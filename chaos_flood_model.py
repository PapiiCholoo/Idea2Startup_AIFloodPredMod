import numpy as np

g = 9.80665

def safe_sqrt(x):
    return np.sqrt(np.maximum(x, 0.0))

class Lorenz63:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, X0=1.0, Y0=1.0, Z0=1.0):
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)
        self.X = float(X0)
        self.Y = float(Y0)
        self.Z = float(Z0)

    def rhs(self, X, Y, Z):
        dX = self.sigma * (Y - X)
        dY = X * (self.rho - Z) - Y
        dZ = X * Y - self.beta * Z
        return dX, dY, dZ

    def step_rk4(self, dt):
        k1x, k1y, k1z = self.rhs(self.X, self.Y, self.Z)
        x2 = self.X + 0.5*dt*k1x
        y2 = self.Y + 0.5*dt*k1y
        z2 = self.Z + 0.5*dt*k1z
        k2x, k2y, k2z = self.rhs(x2, y2, z2)

        x3 = self.X + 0.5*dt*k2x
        y3 = self.Y + 0.5*dt*k2y
        z3 = self.Z + 0.5*dt*k2z
        k3x, k3y, k3z = self.rhs(x3, y3, z3)

        x4 = self.X + dt*k3x
        y4 = self.Y + dt*k3y
        z4 = self.Z + dt*k3z
        k4x, k4y, k4z = self.rhs(x4, y4, z4)

        self.X += dt/6.0*(k1x + 2.0*k2x + 2.0*k3x + k4x)
        self.Y += dt/6.0*(k1y + 2.0*k2y + 2.0*k3y + k4y)
        self.Z += dt/6.0*(k1z + 2.0*k2z + 2.0*k3z + k4z)

    def modulation_factor(self, alpha_P=0.15, sigma_X=7.89):
        X = self.X
        phi = (X / sigma_X) * np.exp(-0.5 * (X / (3.0 * sigma_X))**2)
        lam = 1.0 + alpha_P * phi
        return lam

def willoughby_rahn_r(Pmax_mm_per_h, Rmax_km, r_km, r1_factor=0.5, r2_factor=3.0):
        r1 = r1_factor * Rmax_km
        r2 = r2_factor * Rmax_km
        Pbase_mm_h = Pmax_mm_per_h * (1.0 - np.exp(-r_km / r1)) * np.exp(-r_km / r2)
        Pbase_m_s = (Pbase_mm_h * 1e-3) / 3600.0
        return Pbase_m_s

def compute_typhoon_precip_field(Pmax_mm_h, Rmax_km, x_grid_m, y_grid_m, eye_x_m, eye_y_m):
    dx = (x_grid_m - eye_x_m)
    dy = (y_grid_m - eye_y_m)
    r_km = np.sqrt(dx*dx + dy*dy) / 1000.0
    Pbase = willoughby_rahn_r(Pmax_mm_h, Rmax_km, r_km)
    return Pbase

class SCSInfiltration:
    def __init__(self, CN=70.0):
        self.CN = float(CN)
        S_in = 1000.0 / self.CN - 10.0
        self.S = S_in * 0.0254
        self.Ia = 0.2 * self.S

    def cumulative_runoff_from_P(self, P_cum_m):
        P = P_cum_m
        Q = np.zeros_like(P)
        mask = P > self.Ia
        num = (P[mask] - self.Ia)**2
        den = P[mask] - self.Ia + self.S
        Q[mask] = num / den
        return Q

    def instantaneous_infiltration_rate(self, P_instant_m_s, P_cum_m, dPdt=0.0):
        P = P_cum_m
        P_inst = P_instant_m_s
        I_inst = np.zeros_like(P)
        mask0 = P <= self.Ia
        I_inst[mask0] = P_inst[mask0]
        mask1 = P > self.Ia
        if np.any(mask1):
            Pm = P[mask1]
            dP_dt = dPdt if np.isscalar(dPdt) else dPdt[mask1]
            num = (2.0*(Pm - self.Ia)*(Pm - self.Ia + self.S) - (Pm - self.Ia)**2)
            den = (Pm - self.Ia + self.S)**2
            dQdP = num / den
            dQdt = dQdP * dP_dt
            I_inst[mask1] = dP_dt - dQdt
            I_inst[mask1] = np.maximum(I_inst[mask1], 0.0)
        return I_inst

class FloodGrid:
    def __init__(self, Nx, Ny, Lx, Ly, CN=70.0, manning_n=0.025):
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.xc = (np.arange(Nx) + 0.5) * self.dx
        self.yc = (np.arange(Ny) + 0.5) * self.dy
        X, Y = np.meshgrid(self.xc, self.yc)
        self.X = X
        self.Y = Y
        self.h = np.zeros((Ny, Nx), dtype=float)
        self.u = np.zeros((Ny, Nx), dtype=float)
        self.v = np.zeros((Ny, Nx), dtype=float)
        self.zb = np.zeros((Ny, Nx), dtype=float)
        self.n = manning_n * np.ones((Ny, Nx), dtype=float)
        self.P_cum = np.zeros((Ny, Nx), dtype=float)
        self.Q_cum = np.zeros((Ny, Nx), dtype=float)
        self.CN = CN * np.ones((Ny, Nx), dtype=float)
        self.scs = SCSInfiltration(CN=np.mean(self.CN))
        self.mask = np.ones((Ny, Nx), dtype=float)

    def set_bed_gaussian(self, peak=1966.0, x0=12000.0, y0=8000.0, sigma=3000.0):
        dx = self.X - x0
        dy = self.Y - y0
        self.zb = peak * np.exp(-0.5 * (dx*dx + dy*dy) / (sigma*sigma))

def hllc_flux_1d(hL, uL, vL, hR, uR, vR, g=g):
    tiny = 1e-8
    hL = max(hL, 0.0)
    hR = max(hR, 0.0)
    cL = np.sqrt(g * hL) if hL > tiny else 0.0
    cR = np.sqrt(g * hR) if hR > tiny else 0.0

    SL = min(uL - cL, uR - cR)
    SR = max(uL + cL, uR + cR)

    if SR - SL < 1e-12:
        Fh = 0.5*(hL*uL + hR*uR)
        Fhu = 0.5*(hL*uL*uL + 0.5*g*hL*hL + hR*uR*uR + 0.5*g*hR*hR)
        Fhv = 0.5*(hL*uL*vL + hR*uR*vR)
        return Fh, Fhu, Fhv

    S_star = (SR*uR - SL*uL + 0.5*g*(hL*hL - hR*hR)) / (SR - SL)

    FhL = hL * uL
    FhuL = hL * uL*uL + 0.5 * g * hL*hL
    FhvL = hL * uL * vL

    FhR = hR * uR
    FhuR = hR * uR*uR + 0.5 * g * hR*hR
    FhvR = hR * uR * vR

    if 0.0 <= SL:
        return FhL, FhuL, FhvL
    elif SL < 0.0 <= S_star:
        h_star_L = hL * (SL - uL) / (SL - S_star)
        Fh = FhL + SL * (h_star_L - hL)
        Fhu = FhuL + SL * (h_star_L * S_star - hL * uL)
        Fhv = FhvL + SL * (h_star_L * vL - hL * vL)
        return Fh, Fhu, Fhv
    elif S_star < 0.0 <= SR:
        h_star_R = hR * (SR - uR) / (SR - S_star)
        Fh = FhR + SR * (h_star_R - hR)
        Fhu = FhuR + SR * (h_star_R * S_star - hR * uR)
        Fhv = FhvR + SR * (h_star_R * vR - hR * vR)
        return Fh, Fhu, Fhv
    else:
        return FhR, FhuR, FhvR

def spatial_operator(grid: FloodGrid, g=g):
    Ny, Nx = grid.Ny, grid.Nx
    dx, dy = grid.dx, grid.dy
    h = grid.h
    u = grid.u
    v = grid.v
    zb = grid.zb

    hu = h * u
    hv = h * v

    Fh_x = np.zeros((Ny, Nx+1))
    Fhu_x = np.zeros((Ny, Nx+1))
    Fhv_x = np.zeros((Ny, Nx+1))
    Gh_y = np.zeros((Ny+1, Nx))
    Ghu_y = np.zeros((Ny+1, Nx))
    Ghv_y = np.zeros((Ny+1, Nx))

    h_ext = np.zeros((Ny, Nx+2))
    u_ext = np.zeros((Ny, Nx+2))
    v_ext = np.zeros((Ny, Nx+2))
    h_ext[:, 1:-1] = h
    u_ext[:, 1:-1] = u
    v_ext[:, 1:-1] = v
    zb_ext = np.zeros((Ny, Nx+2))
    zb_ext[:, 1:-1] = zb
    h_ext[:, 0] = h[:, 0]
    u_ext[:, 0] = -u[:, 0]
    v_ext[:, 0] = v[:, 0]
    zb_ext[:, 0] = zb[:, 0]
    h_ext[:, -1] = h[:, -1]
    u_ext[:, -1] = -u[:, -1]
    v_ext[:, -1] = v[:, -1]
    zb_ext[:, -1] = zb[:, -1]

    for i_face in range(Nx+1):
        hl = h_ext[:, i_face]
        ul = u_ext[:, i_face]
        vl = v_ext[:, i_face]
        hr = h_ext[:, i_face+1]
        ur = u_ext[:, i_face+1]
        vr = v_ext[:, i_face+1]
        for j in range(grid.Ny):
            Fh_x[j, i_face], Fhu_x[j, i_face], Fhv_x[j, i_face] = hllc_flux_1d(
                float(hl[j]), float(ul[j]), float(vl[j]),
                float(hr[j]), float(ur[j]), float(vr[j]), g=g)

    h_ext2 = np.zeros((Ny+2, Nx))
    u_ext2 = np.zeros((Ny+2, Nx))
    v_ext2 = np.zeros((Ny+2, Nx))
    zb_ext2 = np.zeros((Ny+2, Nx))
    h_ext2[1:-1, :] = h
    u_ext2[1:-1, :] = u
    v_ext2[1:-1, :] = v
    zb_ext2[1:-1, :] = zb
    h_ext2[0, :] = h[0, :]
    u_ext2[0, :] = u[0, :]
    v_ext2[0, :] = -v[0, :]
    zb_ext2[0, :] = zb[0, :]
    h_ext2[-1, :] = h[-1, :]
    u_ext2[-1, :] = u[-1, :]
    v_ext2[-1, :] = -v[-1, :]
    zb_ext2[-1, :] = zb[-1, :]

    for j_face in range(Ny+1):
        hl = h_ext2[j_face, :]
        ul = u_ext2[j_face, :]
        vl = v_ext2[j_face, :]
        hr = h_ext2[j_face+1, :]
        ur = u_ext2[j_face+1, :]
        vr = v_ext2[j_face+1, :]
        for i in range(grid.Nx):
            Gh, Ghv, Ghu = hllc_flux_1d(
                float(hl[i]), float(vl[i]), float(ul[i]),
                float(hr[i]), float(vr[i]), float(ur[i]), g=g)
            Gh_y[j_face, i] = Gh
            Ghu_y[j_face, i] = Ghu
            Ghv_y[j_face, i] = Ghv

    rhs_h = np.zeros_like(h)
    rhs_hu = np.zeros_like(h)
    rhs_hv = np.zeros_like(h)
    rhs_h += - (Fh_x[:, 1:] - Fh_x[:, :-1]) / dx
    rhs_hu += - (Fhu_x[:, 1:] - Fhu_x[:, :-1]) / dx
    rhs_hv += - (Fhv_x[:, 1:] - Fhv_x[:, :-1]) / dx
    rhs_h += - (Gh_y[1:, :] - Gh_y[:-1, :]) / dy
    rhs_hu += - (Ghu_y[1:, :] - Ghu_y[:-1, :]) / dy
    rhs_hv += - (Ghv_y[1:, :] - Ghv_y[:-1, :]) / dy

    dzbdx = np.zeros_like(zb)
    dzbdy = np.zeros_like(zb)
    dzbdx[:, 1:-1] = (zb[:, 2:] - zb[:, :-2]) / (2.0 * dx)
    dzbdx[:, 0] = (zb[:, 1] - zb[:, 0]) / dx
    dzbdx[:, -1] = (zb[:, -1] - zb[:, -2]) / dx
    dzbdy[1:-1, :] = (zb[2:, :] - zb[:-2, :]) / (2.0 * dy)
    dzbdy[0, :] = (zb[1, :] - zb[0, :]) / dy
    dzbdy[-1, :] = (zb[-1, :] - zb[-2, :]) / dy

    rhs_hu += - g * h * dzbdx
    rhs_hv += - g * h * dzbdy

    return rhs_h, rhs_hu, rhs_hv

def apply_manning_friction(grid: FloodGrid, dt):
        h = grid.h
        hu = h * grid.u
        hv = h * grid.v
        n = grid.n
        tiny = 1e-8
        speed = np.sqrt(grid.u**2 + grid.v**2)
        denom = 1.0 + dt * g * n**2 * speed / (np.maximum(h, tiny)**(4.0/3.0))
        hu_new = hu / denom
        hv_new = hv / denom
        u_new = np.zeros_like(grid.u)
        v_new = np.zeros_like(grid.v)
        mask = grid.h > 1e-8
        u_new[mask] = hu_new[mask] / grid.h[mask]
        v_new[mask] = hv_new[mask] / grid.h[mask]
        grid.u[:, :] = u_new
        grid.v[:, :] = v_new

def apply_sources(grid: FloodGrid, P_total_field_m_s, dt, scs_helper: SCSInfiltration = None):
    if scs_helper is None:
        scs_helper = grid.scs

    dP = P_total_field_m_s * dt
    grid.P_cum += dP
    Q_cum_new = scs_helper.cumulative_runoff_from_P(grid.P_cum)
    dQ = Q_cum_new - grid.Q_cum
    grid.Q_cum = Q_cum_new
    dF = dP - dQ
    I_inst = scs_helper.instantaneous_infiltration_rate(P_total_field_m_s, grid.P_cum, dPdt=P_total_field_m_s)
    grid.h += dP
    grid.h -= dF
    grid.h = np.maximum(grid.h, 0.0)
    apply_manning_friction(grid, dt)

def compute_timestep(grid: FloodGrid, alpha_CFL=0.3, dt_max=1.0, dt_min=1e-6):
    c = np.sqrt(g * np.maximum(grid.h, 0.0))
    lambda_x = np.abs(grid.u) + c
    lambda_y = np.abs(grid.v) + c
    lam_max = np.maximum(np.max(lambda_x), np.max(lambda_y))
    if lam_max <= 0.0:
        return dt_max
    dt1 = alpha_CFL * min(grid.dx, grid.dy) / lam_max
    dt = float(np.clip(dt1, dt_min, dt_max))
    return dt

def pack_state(grid: FloodGrid):
    return np.stack([grid.h, grid.h * grid.u, grid.h * grid.v], axis=0)

def unpack_state(grid: FloodGrid, U):
    grid.h[:, :] = U[0]
    hu = U[1]
    hv = U[2]
    mask = grid.h > 1e-8
    grid.u[:, :] = 0.0
    grid.v[:, :] = 0.0
    grid.u[mask] = hu[mask] / grid.h[mask]
    grid.v[mask] = hv[mask] / grid.h[mask]

def spatial_operator_pack(grid: FloodGrid):
    rhs_h, rhs_hu, rhs_hv = spatial_operator(grid, g=g)
    return np.stack([rhs_h, rhs_hu, rhs_hv], axis=0)

def simulate_flood(grid: FloodGrid, lorenz: Lorenz63, t_end=60.0, output_interval=10.0):
    t = 0.0
    step = 0
    next_output = output_interval
    Pmax_mm_h = 100.0
    Rmax_km = 30.0
    eye_x_m = grid.Lx + 150e3
    eye_y_m = grid.Ly / 2.0

    while t < t_end - 1e-12:
        dt = compute_timestep(grid, alpha_CFL=0.3, dt_max=1.0)
        dt_chaos = 0.01
        n_chaos = max(1, int(np.ceil(dt / dt_chaos)))
        dtc = dt / n_chaos
        for _ in range(n_chaos):
            lorenz.step_rk4(dtc)
        lam = lorenz.modulation_factor(alpha_P=0.15, sigma_X=7.89)
        Pbase = compute_typhoon_precip_field(Pmax_mm_h, Rmax_km, grid.X, grid.Y, eye_x_m, eye_y_m)
        Ptotal = Pbase * lam
        U0 = pack_state(grid)
        rhs0 = spatial_operator_pack(grid)
        U1 = U0 + dt * rhs0
        Utmp = U1.copy()
        unpack_state(grid, Utmp)
        rhs1 = spatial_operator_pack(grid)
        U2 = 0.75 * U0 + 0.25 * (U1 + dt * rhs1)
        unpack_state(grid, U2)
        rhs2 = spatial_operator_pack(grid)
        Unew = (1.0/3.0) * U0 + (2.0/3.0) * (U2 + dt * rhs2)
        unpack_state(grid, Unew)
        apply_sources(grid, Ptotal, dt, scs_helper=grid.scs)
        t += dt
        step += 1
        vtrans = 5.556
        eye_x_m -= vtrans * dt

        if t >= next_output - 1e-9 or t >= t_end - 1e-9:
            print(f"[t={t:.2f}s] step {step}, max h = {grid.h.max():.4f} m")
            next_output += output_interval

    return grid

if __name__ == "__main__":
    Nx, Ny = 40, 20
    Lx, Ly = 20_000.0, 10_000.0
    grid = FloodGrid(Nx, Ny, Lx, Ly, CN=70.0, manning_n=0.03)
    grid.set_bed_gaussian(peak=300.0, x0=8000.0, y0=4000.0, sigma=2000.0)
    cx = int(Nx//2)
    cy = int(Ny//2)
    grid.h[cy-1:cy+2, cx-1:cx+2] = 0.5
    lorenz = Lorenz63(X0=1.0, Y0=1.0, Z0=1.0)
    res = simulate_flood(grid, lorenz, t_end=30.0, output_interval=10.0)
    print("Simulation finished. Max depth:", res.h.max())
