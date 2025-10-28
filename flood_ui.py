import tkinter as tk
from tkinter import ttk, messagebox
import threading
import matplotlib.pyplot as plt

import chaos_flood_model as cfm

class FloodUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaos-Enhanced Flood Prediction System")
        self.root.geometry("520x600")

        ttk.Label(root, text="Chaos-Enhanced Flood Prediction System", font=("Arial", 14, "bold")).pack(pady=10)

        frame = ttk.Frame(root)
        frame.pack(padx=15, pady=5, fill="x")

        self.entries = {}
        params = [
            ("Grid Size Nx", 40),
            ("Grid Size Ny", 20),
            ("Domain Length Lx (m)", 20000),
            ("Domain Width Ly (m)", 10000),
            ("Curve Number CN", 70),
            ("Manning's n", 0.03),
            ("Simulation Time (s)", 30),
            ("Output Interval (s)", 10),
            ("Lorenz σ", 10),
            ("Lorenz ρ", 28),
            ("Lorenz β", 8/3),
        ]

        for label, default in params:
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=22).pack(side="left")
            e = ttk.Entry(row)
            e.insert(0, str(default))
            e.pack(side="right", expand=True, fill="x")
            self.entries[label] = e

        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Run Simulation", command=self.run_simulation_thread).pack()

        ttk.Label(root, text="Simulation Log:").pack(anchor="w", padx=15)
        self.text = tk.Text(root, height=15, bg="#111", fg="#0f0", insertbackground="white")
        self.text.pack(padx=15, pady=5, fill="both", expand=True)

    def log(self, msg):
        self.text.insert(tk.END, msg + "\n")
        self.text.see(tk.END)
        self.root.update_idletasks()

    def run_simulation_thread(self):
        thread = threading.Thread(target=self.run_simulation)
        thread.start()

    def run_simulation(self):
        try:
            Nx = int(self.entries["Grid Size Nx"].get())
            Ny = int(self.entries["Grid Size Ny"].get())
            Lx = float(self.entries["Domain Length Lx (m)"].get())
            Ly = float(self.entries["Domain Width Ly (m)"].get())
            CN = float(self.entries["Curve Number CN"].get())
            n = float(self.entries["Manning's n"].get())
            t_end = float(self.entries["Simulation Time (s)"].get())
            output_interval = float(self.entries["Output Interval (s)"].get())
            sigma = float(self.entries["Lorenz σ"].get())
            rho = float(self.entries["Lorenz ρ"].get())
            beta = float(self.entries["Lorenz β"].get())

            self.log("Initializing simulation grid...")
            grid = cfm.FloodGrid(Nx, Ny, Lx, Ly, CN=CN, manning_n=n)
            grid.set_bed_gaussian(peak=300.0, x0=Lx/2, y0=Ly/2, sigma=0.25*Lx)

            lorenz = cfm.Lorenz63(sigma=sigma, rho=rho, beta=beta)
            self.log("Starting simulation...")

            import sys
            old_stdout = sys.stdout
            sys.stdout = self
            self.buffer = ""

            result = cfm.simulate_flood(grid, lorenz, t_end=t_end, output_interval=output_interval)

            sys.stdout = old_stdout

            self.log("Simulation complete.")
            self.log(f"Max depth: {result.h.max():.4f} m")

            self.log("Plotting results...")
            plt.figure(figsize=(6, 4))
            plt.imshow(result.h, cmap="Blues", origin="lower", extent=[0, Lx/1000, 0, Ly/1000])
            plt.colorbar(label="Water Depth (m)")
            plt.title("Final Flood Depth Distribution")
            plt.xlabel("X (km)")
            plt.ylabel("Y (km)")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def write(self, s):
        if s.strip():
            self.log(s.strip())

    def flush(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = FloodUI(root)
    root.mainloop()
