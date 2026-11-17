"""Transient 3-D TensorFlow PINN for a homogenized, sealed vapor chamber.

Coordinates are (x/L, y/W, z/H, t/t_end); all configuration values use SI.
The calibrated anisotropic heat operator already represents latent transport.
Heat drives a reduced two-phase circulation diagnostic; its inferred latent
power is not added a second time to the thermal equation. No bubble radius,
moving interface, local adsorption, dry-out front, or chip junction is solved.
"""
import math
import numpy as np
import tensorflow as tf


def _input(points):
    points = tf.cast(tf.convert_to_tensor(points), tf.float32)
    return tf.ensure_shape(points, [None, 4])


def scalar_jet(fn, points):
    """Value, gradient wrt all 4 coordinates, and three pure spatial derivatives.

    Rows must be independent (no batch normalization or other cross-row layers).
    A missing first derivative is an error; a mathematically constant first
    derivative of a linear field has zero curvature. Slice within tape scope.
    """
    points = _input(points)
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as outer:
        outer.watch(points)
        with tf.GradientTape(watch_accessed_variables=False) as inner:
            inner.watch(points)
            value = fn(points)
        first = inner.gradient(value, points)
        if first is None:
            raise ValueError("Scalar field is disconnected from coordinates.")
        parts = tf.split(first, 4, axis=1)[:3]
    second = [
        outer.gradient(part, points, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, i:i+1]
        for i, part in enumerate(parts)
    ]
    del outer
    return value, first, tf.concat(second, axis=1)


def energy_residual(first, hdiag, cfg):
    """Dimensionless effective heat equation, divided by its diffusion scale."""
    fo = [cfg.t_end * k / (cfg.rho_cp * length**2)
          for k, length in zip((cfg.k_xy, cfg.k_xy, cfg.k_z), (cfg.L, cfg.W, cfg.H))]
    return (first[:, 3:4] - tf.reduce_sum(hdiag * fo, axis=1, keepdims=True)) / max(1., sum(fo))


def _positive_part_square(x):
    return tf.square(tf.nn.relu(x))


def _smooth_patch(x, y, rect):
    """Differentiable footprint only for the initial guess, never for applied W."""
    xa, xb, ya, yb = rect
    def window(coord, a, b):
        edge = 0.08 * (b - a)
        return 0.5 * (tf.tanh((coord-a)/edge) - tf.tanh((coord-b)/edge))
    return window(x, xa, xb) * window(y, ya, yb)


def _unit_gaussian(x, center, sd):
    """Gaussian density restricted to [0, 1], with integral exactly one."""
    den = sd * math.sqrt(math.pi/2.) * (
        math.erf((1.-center)/(math.sqrt(2.)*sd)) + math.erf(center/(math.sqrt(2.)*sd)))
    return tf.exp(-0.5*tf.square((x-center)/sd)) / den


def _wall_coordinate(x, band=.08):
    """Monotone coordinate, identical in the interior, zero slope at both ends."""
    return (x - x*tf.nn.relu(1.-x/band)**3
            + (1.-x)*tf.nn.relu(1.-(1.-x)/band)**3)


def _wall_location(x, band=.08):
    return x-x*max(0.,1.-x/band)**3+(1.-x)*max(0.,1.-(1.-x)/band)**3


def _quadrature_rule():
    """Normalized cell Gauss points followed by six face quadratures."""
    x, w = np.polynomial.legendre.leggauss(3)
    x, w = (x+1.)/2., w/2.
    volume = np.stack(np.meshgrid(x,x,x,indexing="ij"),axis=-1).reshape(-1,3)
    vw = np.prod(np.meshgrid(w,w,w,indexing="ij"),axis=0).ravel()
    uv = np.stack(np.meshgrid(x,x,indexing="ij"),axis=-1).reshape(-1,2)
    fw = np.prod(np.meshgrid(w,w,indexing="ij"),axis=0).ravel()
    sites = [volume]
    for axis in range(3):
        for face in (0.,1.):
            sites.append(np.insert(uv,axis,face,axis=1))
    return tuple(tf.constant(v,tf.float32) for v in (np.concatenate(sites),vw,fw))


class PhysicsModel(tf.keras.Model):
    """Dense smooth network plus dimensional physics and sealed-wall constraints."""

    face_keys = ("x0", "x1", "y0", "y1", "z0", "z1", "cpu", "gpu")

    def __init__(self, case, width=32, depth=3):
        super().__init__()
        self.cfg = c = case
        self.hidden = [tf.keras.layers.Dense(width, activation="tanh") for _ in range(depth)]
        # θ correction, ux,uy,uz liquid and vapor, liquid/vapor gauge, solute.
        self.out = tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=.025))
        self.lengths = (c.L, c.W, c.H)
        self.anis = [a/c.L for a in self.lengths]
        self.Ul = c.power / (c.rho*c.h_fg*c.fill*c.void_fraction*c.W*c.H)
        self.Uv = c.power / (c.rho_v*c.h_fg*(1.-c.fill)*c.void_fraction*c.W*c.H)
        self.P_l = c.mu * self.Ul * c.L / c.permeability
        self.P_v = c.mu_v * self.Uv * c.L / c.vapor_permeability
        self.watts = tuple(q*(xb-xa)*c.L*(yb-ya)*c.W
                           for q, (xa,xb,ya,yb) in zip((c.q_cpu,c.q_gpu), (c.cpu,c.gpu)))
        self.capacity = c.rho_cp * c.volume
        area_end = c.W*c.H
        side_ua = 2*c.h_side*area_end
        faces_ua = c.h_z*(2*c.L*c.W - sum(w/q for w,q in zip(self.watts,(c.q_cpu,c.q_gpu)) if q > 0))
        ua = side_ua + faces_ua
        if ua <= 0:
            raise ValueError("Positive heat rejection conductance is required.")
        self.side_fraction = side_ua/ua
        # One-dimensional conducting strip supplies a useful differentiable guess.
        centers = [(r[0]+r[1])/2 for r in (c.cpu,c.gpu)]
        self.left_w = ((c.power*c.k_xy + c.h_side*c.L*sum(w*(1-x) for w,x in zip(self.watts,centers)))
                       / (2*c.k_xy + c.h_side*c.L)) * self.side_fraction
        if c.h_side > 0:
            self.end_rise = self.left_w/(area_end*c.h_side)
        else:
            self.end_rise = c.power/ua
        # Integrate the steady 1-D guess analytically to set its startup time.
        mean = self.end_rise + self.left_w*c.L/(2*c.k_xy*area_end)
        for w,(xa,xb,_,_) in zip(self.watts,(c.cpu,c.gpu)):
            mean -= self.side_fraction*w*c.L*((1-xa)**3-(1-xb)**3)/(6*c.k_xy*area_end*(xb-xa))
        self.tau = max(1.e-6, self.capacity*mean/c.power)
        self.tau_z = c.rho_cp*c.H**2/c.k_z
        self.temperature_bias = float(np.log(np.expm1(c.T_amb/c.dT))) if c.T_amb/c.dT < 40 else c.T_amb/c.dT
        self._quad_sites, self._quad_vol, self._quad_face = _quadrature_rule()
        self._reference_flux = c.power/(2*area_end)
        self._z_flux = max(c.h_z*c.dT, .15*max(c.q_cpu,c.q_gpu))

    def call(self, points):
        # No BatchNorm: gradient-of-sum computes each row's own derivatives.
        c = self.cfg
        x,y,z,t = tf.split(points,4,axis=1)
        y,z = _wall_coordinate(y),_wall_coordinate(z)
        # Smooth, geometry-specific local and early-time features. They allow a
        # compact tanh net to distinguish a die edge from a cooler neighboring
        # point. The actual boundary samples still apply the exact die area.
        features = [2.*tf.concat([x,y,z,t],axis=1)-1., -tf.math.expm1(-t*c.t_end/self.tau),
                    -tf.math.expm1(-t*c.t_end/self.tau_z)]
        for xa,xb,ya,yb in (c.cpu,c.gpu):
            dx = (x-.5*(xa+xb))/(.55*(xb-xa))
            dy = (y-.5*(ya+yb))/(.55*(yb-ya))
            features.append(tf.exp(-.5*(dx*dx+dy*dy)))
        features = tf.concat(features,axis=1)
        for layer in self.hidden:
            features = layer(features)
        return self.out(features)

    def _theta(self, pts, raw):
        c = self.cfg
        x,y,z,s = tf.split(pts, 4, axis=1)
        y_wall, z_wall = _wall_coordinate(y), _wall_coordinate(z)
        rise = self.end_rise + self.left_w*x*c.L/(c.k_xy*c.W*c.H)
        wall = tf.zeros_like(x)
        # Extension of a step-flux footprint into the slab. Its smoothing width
        # grows with physical distance from z=0; the wall value itself uses the
        # exact die rectangle (the discontinuity curve has zero surface area).
        dx = tf.maximum(1e-6, z*c.H/c.L*math.sqrt(c.k_xy/c.k_z))
        dy = tf.maximum(1e-6, z*c.H/c.W*math.sqrt(c.k_xy/c.k_z))
        heat_flux = tf.zeros_like(x)
        covered = tf.zeros_like(x)
        def window(coord, a, b, w):
            return .5*(tf.tanh((coord-a)/w)-tf.tanh((coord-b)/w))
        for watts, q, rect in zip(self.watts, (c.q_cpu,c.q_gpu), (c.cpu,c.gpu)):
            xa,xb,ya,yb = rect
            rise -= (self.side_fraction*watts*c.L/(2*c.k_xy*c.W*c.H*(xb-xa))
                     * (_positive_part_square(x-xa) - _positive_part_square(x-xb)))
            ya_w,yb_w = _wall_location(ya),_wall_location(yb)
            wall += ((q*c.H/(2*c.k_z))*tf.square(1.-z_wall)
                     * _smooth_patch(x,y_wall,(xa,xb,ya_w,yb_w)))
            smooth = window(x,xa,xb,dx)*window(y_wall,ya_w,yb_w,dy)
            exact = (x>=xa)&(x<=xb)&(y>=ya)&(y<=yb)
            mask = tf.where(z==0.,tf.cast(exact,pts.dtype),smooth)
            covered += mask
            heat_flux += q*mask
        g_bulk = -tf.math.expm1(-s*c.t_end/self.tau)
        g_wall = -tf.math.expm1(-s*c.t_end/self.tau_z)
        delta = (g_bulk*rise + g_wall*wall)/c.dT + g_wall*raw[:,0:1]
        bias = tf.cast(self.temperature_bias, pts.dtype)
        absolute = tf.nn.softplus(bias+delta)
        theta = absolute - tf.nn.softplus(bias)
        # For t>0 the z=0 derivative of the lifting term is exactly -qH/k_z
        # on each energized die. At t=0 the entire lift is zero. Its switch is
        # continuous for interior points and resolves the incompatible startup
        # corner of an instant power step without prescribing a temperature.
        switch = tf.math.divide_no_nan(s*c.t_end,s*c.t_end+self.tau_z*z*z)
        lifting = -heat_flux*c.H/(c.k_z*c.dT)*z*(1.-z)**2*switch
        # Free z=0: dtheta/dz=Bi*theta. z=1: dtheta/dz=-Bi*theta.
        robin = c.h_z*c.H/c.k_z * z*(1.-z)*(1.-(1.-z)*covered)*theta
        # Exponential construction preserves positive absolute temperature while
        # exactly reproducing the prescribed derivative on each z boundary.
        shift = tf.math.divide_no_nan(lifting+robin,absolute)
        return theta + absolute*tf.math.expm1(shift)

    def _solute(self, pts, raw):
        gate = -tf.math.expm1(-4.*pts[:,3:4])
        # Relative concentration is positive; true mass fraction c=c0*rel.
        return tf.nn.softplus(gate*raw[:,9:10]) / math.log(2.)

    def temperature_C(self, points):
        pts = _input(points)
        return self.cfg.T_amb - 273.15 + self.cfg.dT*self._theta(pts, self(pts))

    def _states(self, pts):
        raw = self(pts)
        xyz, s = pts[:,:3], pts[:,3:4]
        # Gauge at spatial center for each time. Absolute vapor pressure is not
        # identifiable from this incompressible hydrodynamic closure.
        ref = self(tf.concat([tf.fill(tf.shape(xyz), .5), s], axis=1))
        gate = -tf.math.expm1(-4.*s)
        envelope = 4.*xyz*(1.-xyz)
        liquid = raw[:,1:4]*tf.reduce_prod(envelope,axis=1,keepdims=True)*gate
        vapor = raw[:,4:7]*envelope*gate
        pressure = (raw[:,7:9]-ref[:,7:9])*gate
        return (self._theta(pts,raw), *tf.split(liquid,3,axis=1), *tf.split(vapor,3,axis=1),
                *tf.split(pressure,2,axis=1), self._solute(pts,raw))

    def _phase(self, pts):
        """Signed kg/(m³ s), kinetic saturation proxy °C and kg/s amplitude.

        Equal evaporator/condenser conductances G=h_i*a_i*V/2 in series.
        Four point thermometers approximate the two reservoir temperatures.
        The volume integral is zero at every t: both spatial densities have
        unit integral. This fixed-interface closure does not resolve a Stefan
        jump, bubbles, nucleation, dynamic surface tension, or a dry-out front.
        """
        c = self.cfg
        x,y,z,s = tf.split(pts,4,axis=1)
        centers = [((a+b)/2,(d+e)/2) for a,b,d,e in (c.cpu,c.gpu)]
        sensors = [(cx,cy,.25) for cx,cy in centers] + [(0.,.5,.5),(1.,.5,.5)]
        blocks = [tf.concat([tf.ones_like(s)*v for v in point] + [s],axis=1) for point in sensors]
        cpu_t,gpu_t,left_t,right_t = tf.split(self.temperature_C(tf.concat(blocks,axis=0)),4,axis=0)
        hot = (self.watts[0]*cpu_t + self.watts[1]*gpu_t)/c.power
        cold = (left_t+right_t)/2.
        mdot = (hot-cold)*c.interface_h_vol*c.volume/(4.*c.h_fg)
        rate = mdot/c.volume*self.phase_profile(pts)
        return rate,(hot+cold)/2.,mdot

    def phase_profile(self, points):
        """Zero-mean dimensionless source-density shape, shared by FV/PINN."""
        pts = _input(points)
        c = self.cfg
        x,y,z,_ = tf.split(pts,4,axis=1)
        centers = [((a+b)/2,(d+e)/2) for a,b,d,e in (c.cpu,c.gpu)]
        evaporation = tf.zeros_like(x)
        for w,rect,(cx,cy) in zip(self.watts,(c.cpu,c.gpu),centers):
            evaporation += (w/c.power * _unit_gaussian(x,cx,.35*(rect[1]-rect[0]))
                            * _unit_gaussian(y,cy,.35*(rect[3]-rect[2])))
        condensation = .5*(_unit_gaussian(x,0.,.06) + _unit_gaussian(x,1.,.06))
        return 6.*z*(1.-z)*(evaporation-condensation)

    def fields(self, points):
        pts = _input(points)
        c = self.cfg
        theta,u,v,w,U,V,W,p,pv,solute = self._states(pts)
        j,ts,mdot = self._phase(pts)
        return dict(T_C=c.T_amb-273.15+c.dT*theta,
                    u_m_s=self.Ul*u, v_m_s=self.Ul*self.anis[1]*v, w_m_s=self.Ul*self.anis[2]*w,
                    p_Pa=self.P_l*p, c_wt_pct=100.*c.c0*solute,
                    u_vapor_m_s=self.Uv*U,v_vapor_m_s=self.Uv*self.anis[1]*V,
                    w_vapor_m_s=self.Uv*self.anis[2]*W,p_vapor_gauge_Pa=self.P_v*pv,
                    phase_rate_kg_m3_s=j,T_saturation_proxy_C=ts,phase_circulation_kg_s=mdot)

    predict_fields = fields

    def solute_transport(self, points):
        """Dimensional nonvolatile-solute flux and continuum flux velocity.

        J = φ_l ρ_l (c u_l − D ∇c) is kg/(gross m² s). The flux velocity
        u_c = u_l − D ∇ln(c) is a dissolved-species mean, not a particle
        trajectory. With DI water J=0 and the solute velocity is undefined.
        """
        pts = _input(points)
        c = self.cfg
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(pts)
            state = self._states(pts)
            relative = state[9]
        drel = tape.gradient(relative,pts)[:,:3] / self.lengths
        liquid = tf.concat(state[1:4],axis=1) * self.anis * self.Ul
        mass_flux = c.void_fraction*c.fill*c.rho*c.c0 * (relative*liquid - c.D*drel)
        result = {"mass_flux_kg_m2_s":mass_flux}
        if c.c0 > 0.:
            result["solute_velocity_m_s"] = liquid - c.D*drel/relative
        return result

    def residuals(self, points, include_energy=True):
        """Per-row nondimensional residuals; +j evaporates mass out of liquid.

        Choose liquid and vapor speeds from P/(h_fg*rho*area*volume fraction).
        Axis velocities scale as U_i=U*length_i/L. Both continuity equations
        then scale by U/L. Pressure scales are μUL/K for each phase. Generalized
        incompressible liquid NS adds viscous diffusion and Darcy drag; with
        constant intrinsic density and mass transfer carrying local velocity,
        conservative momentum reduces to this acceleration form. Vapor uses
        quasi-steady Darcy transport. Solute uses conservative advection and
        diffusion, including c div(u), with zero solute in vapor.
        """
        c = self.cfg
        pts = _input(points)
        curved = [1,2,3]
        if include_energy:
            curved.insert(0,0)
        if c.c0 > 0:
            curved.append(9)
        with tf.GradientTape(persistent=True,watch_accessed_variables=False) as second:
            second.watch(pts)
            with tf.GradientTape(persistent=True,watch_accessed_variables=False) as first:
                first.watch(pts)
                state = self._states(pts)
            gradients = [first.gradient(field,pts) if k!=0 or include_energy else None
                         for k,field in enumerate(state)]
            del first
            if any(g is None for g in gradients[0 if include_energy else 1:]):
                raise ValueError("A physical state is disconnected from its coordinates.")
            parts = {k:tf.split(gradients[k],4,axis=1)[:3] for k in curved}
        hess = {k:tf.concat([
            second.gradient(part,pts,unconnected_gradients=tf.UnconnectedGradients.ZERO)[:,i:i+1]
            for i,part in enumerate(parts[k])],axis=1) for k in curved}
        del second
        r = {"energy": energy_residual(gradients[0],hess[0],c)} if include_energy else {}
        j,_,_ = self._phase(pts)
        # With the chosen scales both phases have the same mass-rate scale.
        jstar = j*c.volume*c.h_fg/c.power
        div_l = tf.add_n([gradients[i+1][:,i:i+1] for i in range(3)])
        div_v = tf.add_n([gradients[i+4][:,i:i+1] for i in range(3)])
        r["continuity_liquid"] = div_l+jstar
        r["continuity_vapor"] = div_v-jstar
        vel = tf.concat(state[1:4],axis=1)
        transient = c.rho*c.permeability/(c.mu*c.t_end)
        inertia = c.rho*self.Ul*c.permeability/(c.mu*c.L)
        brink = [c.permeability/a**2 for a in self.lengths]
        for i,axis in enumerate("xyz"):
            k = i+1
            g = gradients[k]
            a = self.anis[i]
            # Normalize by the pressure-gradient force scale P_ref/length_i.
            accel = transient*g[:,3:4] + inertia*tf.reduce_sum(vel*g[:,:3],axis=1,keepdims=True)
            visc = tf.reduce_sum(hess[k]*brink,axis=1,keepdims=True)
            r["momentum_liquid_"+axis] = a*a*(accel+state[k]-visc) + gradients[7][:,i:i+1]
            r["darcy_vapor_"+axis] = a*a*state[i+4] + gradients[8][:,i:i+1]
        if c.c0 > 0.:
            transport = self.Ul*c.t_end/c.L
            diff = [c.D*c.t_end/a**2 for a in self.lengths]
            flux_div = tf.reduce_sum(vel*gradients[9][:,:3],axis=1,keepdims=True) + state[9]*div_l
            r["solute"] = ((gradients[9][:,3:4] + transport*flux_div
                            - tf.reduce_sum(hess[9]*diff,axis=1,keepdims=True))
                           / max(1.,transport,sum(diff)))
        return r

    def boundary_residuals(self, points):
        """Fluxes use outward signs; CPU/GPU specify inward watts per die area."""
        c = self.cfg
        faces = [_input(points[key]) for key in self.face_keys]
        sizes = tf.stack([tf.shape(f)[0] for f in faces])
        pts = tf.concat(faces,axis=0)
        with tf.GradientTape(persistent=True,watch_accessed_variables=False) as tape:
            tape.watch(pts)
            raw = self(pts)
            theta = self._theta(pts,raw)
            rel = self._solute(pts,raw)
        g = tape.gradient(theta,pts)
        gs = tape.gradient(rel,pts) if c.c0 > 0 else None
        del tape
        theta_parts = tf.split(theta,sizes,axis=0)
        g_parts = tf.split(g,sizes,axis=0)
        if gs is not None:
            s_parts = tf.split(gs,sizes,axis=0)
        out = {}
        for i,(key,axis,normal) in enumerate(zip(self.face_keys,(0,0,1,1,2,2,2,2),(-1,1,-1,1,-1,1,-1,-1))):
            k = (c.k_xy,c.k_xy,c.k_z)[axis]
            flux = -normal*k*c.dT/self.lengths[axis] * g_parts[i][:,axis:axis+1]
            if key in ("cpu","gpu"):
                q = c.q_cpu if key=="cpu" else c.q_gpu
                resid = (flux+q)/max(q,1.)
            elif axis == 1:
                resid = flux / (k*c.dT/c.W)
            else:
                h, scale = (c.h_side,self._reference_flux) if axis==0 else (c.h_z,self._z_flux)
                resid = (flux-h*c.dT*theta_parts[i])/scale
            out["heat_"+key] = resid
            if gs is not None:
                out["solute_wall_"+key] = s_parts[i][:,axis:axis+1]
        return out

    def control_volume_energy(self, points):
        """Weak heat balance in W on local boxes and the full chamber.

        The discontinuous die heat flux has sharp edge curvature, where a
        pointwise squared second-derivative residual can dominate a resampled
        minibatch. For training integrate dU/dt + Q_out on independent boxes.
        Use product Gauss quadrature, natural exterior BCs and *exact* die
        overlap. Interior surface flux is the conductive gradient. Local
        error is a fraction of the applied power scaled by relative box area
        (normalized box volume to power 2/3); global error is divided by P.
        The differential heat residual is retained by residuals() for audits.
        """
        c = self.cfg
        pts = _input(points)
        sampled = pts[:6]
        n = tf.shape(sampled)[0]
        sizes = tf.gather(tf.constant([[.24,.34,.42],[.46,.58,.70],[.65,.52,.42]]),
                          tf.range(n)%3)
        raw_lower = sampled[:,:3] - sizes/2.
        lo = tf.clip_by_value(raw_lower,0.,1.-sizes)
        hi = lo + sizes
        # Test full contact power through two finite, footprint-aligned prisms.
        die_lo = [[r[0],r[2],0.] for r in (c.cpu,c.gpu)]
        die_hi = [[r[1],r[3],.50] for r in (c.cpu,c.gpu)]
        lo = tf.concat([lo,die_lo,tf.zeros((2,3))],axis=0)
        hi = tf.concat([hi,die_hi,tf.ones((2,3))],axis=0)
        more_s = tf.gather(pts[:,3:4],tf.range(4)%tf.shape(pts)[0])
        # One whole-chamber time is biased to startup where storage matters.
        more_s = tf.concat([more_s[:3],tf.square(more_s[3:4])],axis=0)
        times = tf.concat([sampled[:,3:4],more_s],axis=0)
        extent = hi-lo
        xyz = lo[:,None,:] + extent[:,None,:]*self._quad_sites[None,:,:]
        st = tf.broadcast_to(times[:,None,:],tf.shape(xyz[:,:,:1]))
        X = tf.reshape(tf.concat([xyz,st],axis=2),(-1,4))
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(X)
            theta = self._theta(X,self(X))
        grad = tf.reshape(tape.gradient(theta,X),(-1,81,4))
        theta = tf.reshape(theta,(-1,81))
        vol_frac = tf.reduce_prod(extent,axis=1)
        heat = (c.rho_cp*c.volume*vol_frac*c.dT/c.t_end
                * tf.reduce_sum(grad[:,:27,3]*self._quad_vol,axis=1))
        for face in range(6):
            axis, side = divmod(face,2)
            i = 27+9*face
            boundary = lo[:,axis]<1e-6 if side==0 else hi[:,axis]>1.-1e-6
            conductivity = (c.k_xy,c.k_xy,c.k_z)[axis]
            flux = (-1 if side else 1)*conductivity*c.dT/self.lengths[axis]*tf.reduce_sum(
                grad[:,i:i+9,axis]*self._quad_face,axis=1)
            area = c.volume/self.lengths[axis]*vol_frac/extent[:,axis]
            h = (c.h_side,0.,c.h_z)[axis]
            face_t = theta[:,i:i+9]*c.dT
            imposed_W = tf.zeros_like(area)
            if face == 4:
                outside = tf.ones_like(face_t)
                xy = xyz[:,i:i+9,:2]
                for q,(xa,xb,ya,yb) in zip((c.q_cpu,c.q_gpu),(c.cpu,c.gpu)):
                    dx = tf.nn.relu(tf.minimum(hi[:,0],xb)-tf.maximum(lo[:,0],xa))
                    dy = tf.nn.relu(tf.minimum(hi[:,1],yb)-tf.maximum(lo[:,1],ya))
                    imposed_W -= q*c.L*c.W*dx*dy
                    inside = (xy[:,:,0]>=xa)&(xy[:,:,0]<=xb)&(xy[:,:,1]>=ya)&(xy[:,:,1]<=yb)
                    outside *= 1.-tf.cast(inside,tf.float32)
                face_t *= outside
            exterior_W = imposed_W + h*area*tf.reduce_sum(face_t*self._quad_face,axis=1)
            heat += tf.where(boundary,exterior_W,area*flux)
        local = heat[:-2]/(c.power*vol_frac[:-2]**(2./3.))
        return local[:,None], heat[-2:,None]/c.power

    def losses_for(self, points):
        """MSE by equation and face. The IC is embedded, sampled for verification."""
        residuals = dict(self.residuals(points["interior"],include_energy=False),
                         **self.boundary_residuals(points))
        local, chamber = self.control_volume_energy(points["interior"])
        residuals.update(energy_control_volume=local, energy_global=chamber)
        losses = {k:tf.reduce_mean(tf.square(v)) for k,v in residuals.items()}
        if "initial" in points:
            state = self._states(_input(points["initial"]))
            initial_errors = [state[0],*state[1:9],state[9]-1.]
            losses["initial"] = tf.reduce_mean(tf.square(tf.concat(initial_errors,axis=1)))
        return losses
