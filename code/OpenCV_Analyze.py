import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_and_compare():
  files = {
      "Phase 1 (Baseline CPU)": "phase1_telemetry_detailed.csv",
      "Phase 2 (GPU Transfer)": "phase2_telemetry_detailed.csv",
      "Phase 3 (AI-Upscaling)": "phase3_telemetry_detailed.csv",
      "Phase 4 (ESPCN AI-Upscaling)": "phase4_espcn_telemetry_detailed.csv",
  }

  summary_data = []
  dfs = {}

  print("\n" + "=" * 90)
  print("🔬 TIEFGEHENDE SYSTEM-ANALYSE & PHASENVERGLEICH")
  print("=" * 90)

  for name, path in files.items():
    if not os.path.exists(path):
      print(f"⚠️ Warnung: '{path}' nicht gefunden. Überspringe...")
      continue

    df = pd.read_csv(path)
    dfs[name] = df

    # Statistische Kennzahlen berechnen
    avg_fps = df["Instant_FPS"].mean()
    std_fps = df["Instant_FPS"].std()  # FPS-Jitter / Stabilität
    p5_fps = np.percentile(df["Instant_FPS"], 5)  # 5% Lows (Stotter-Indikator)

    peak_vram = df["VRAM_MB"].max()
    avg_vram = df["VRAM_MB"].mean()
    peak_ram = df["RAM_MB"].max()
    max_temp = df["GPU_Temp_C"].max()
    avg_load = df["GPU_Load_Percent"].mean()

    # Latenz ermitteln (je nach Phase)
    latency_col = None
    if "Transfer_Latency_ms" in df.columns:
      latency_col = "Transfer_Latency_ms"
    elif "Inference_Latency_ms" in df.columns:
      latency_col = "Inference_Latency_ms"

    avg_latency = df[latency_col].mean() if latency_col and latency_col in df.columns else 0.0
    max_latency = df[latency_col].max() if latency_col and latency_col in df.columns else 0.0

    summary_data.append({
        "Phase": name,
        "Avg FPS": round(avg_fps, 2),
        "FPS StdDev (Jitter)": round(std_fps, 2),
        "5% Lows FPS": round(p5_fps, 2),
        "Lat Ø (ms)": round(avg_latency, 2),
        "Lat Max (ms)": round(max_latency, 2),
        "Peak VRAM (MB)": round(peak_vram, 2),
        "Peak RAM (MB)": round(peak_ram, 2),
        "Max Temp (°C)": round(max_temp, 1),
        "GPU Load (%)": round(avg_load, 1),
    })

  if not summary_data:
    print("❌ Keine CSV-Dateien gefunden!")
    return

  summary_df = pd.DataFrame(summary_data)

  # Terminal-Ausgabe
  print(summary_df.to_string(index=False))
  print("=" * 90 + "\n")

  # Zusammenfassung als CSV exportieren
  summary_csv = "comprehensive_summary_report.csv"
  summary_df.to_csv(summary_csv, index=False)
  print(f"📁 Tabellarische Zusammenfassung gespeichert unter: {summary_csv}")

  # --- ERWEITERTER 4ER-VERGLEICHS-PLOT ---
  fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=300)

  # 1. FPS Verlauf
  axs[0, 0].set_title(
      "Bildraten-Stabilität (Instant FPS)", fontweight="bold"
  )
  for name, df in dfs.items():
    axs[0, 0].plot(
        df["Timestamp"], df["Instant_FPS"], label=name, alpha=0.7, linewidth=1.2
    )
  axs[0, 0].set_ylabel("FPS")
  axs[0, 0].grid(True, linestyle=":", alpha=0.6)
  axs[0, 0].legend(loc="lower left")

  # 2. Latenz-Vergleich (Transfer vs. Inferenz für alle Phasen)
  axs[0, 1].set_title(
      "Frame-Verarbeitungslatenz (Transfer / Inferenz)", fontweight="bold"
  )
  if "Phase 2 (GPU Transfer)" in dfs:
    df2 = dfs["Phase 2 (GPU Transfer)"]
    if "Transfer_Latency_ms" in df2.columns:
      axs[0, 1].plot(
          df2["Timestamp"],
          df2["Transfer_Latency_ms"],
          label="Phase 2 Transfer",
          color="dodgerblue",
      )
  if "Phase 3 (AI-Upscaling)" in dfs:
    df3 = dfs["Phase 3 (AI-Upscaling)"]
    if "Inference_Latency_ms" in df3.columns:
      axs[0, 1].plot(
          df3["Timestamp"],
          df3["Inference_Latency_ms"],
          label="Phase 3 Inferenz",
          color="crimson",
      )
  if "Phase 4 (ESPCN AI-Upscaling)" in dfs:
    df4 = dfs["Phase 4 (ESPCN AI-Upscaling)"]
    if "Inference_Latency_ms" in df4.columns:
      axs[0, 1].plot(
          df4["Timestamp"],
          df4["Inference_Latency_ms"],
          label="Phase 4 ESPCN Inferenz",
          color="darkorange",
          linestyle="--",
      )
  axs[0, 1].set_ylabel("Latenz in ms")
  axs[0, 1].grid(True, linestyle=":", alpha=0.6)
  axs[0, 1].legend(loc="upper right")

  # 3. VRAM Footprint
  axs[1, 0].set_title("VRAM-Auslastung (GPU Memory)", fontweight="bold")
  for name, df in dfs.items():
    axs[1, 0].plot(
        df["Timestamp"], df["VRAM_MB"], label=name, linewidth=2, alpha=0.9
    )
  axs[1, 0].set_xlabel("Zeit in Sekunden")
  axs[1, 0].set_ylabel("VRAM (MB)")
  axs[1, 0].grid(True, linestyle=":", alpha=0.6)
  axs[1, 0].legend(loc="upper left")

  # 4. GPU Temperatur & Last
  axs[1, 1].set_title(
      "Thermales Verhalten & GPU Core Load", fontweight="bold"
  )
  for name, df in dfs.items():
    axs[1, 1].plot(
        df["Timestamp"],
        df["GPU_Temp_C"],
        label=f"{name} (Temp)",
        linestyle="-",
    )
  axs[1, 1].set_xlabel("Zeit in Sekunden")
  axs[1, 1].set_ylabel("Temperatur (°C)")
  axs[1, 1].grid(True, linestyle=":", alpha=0.6)
  axs[1, 1].legend(loc="upper left")

  plt.tight_layout()
  output_plot = "comprehensive_benchmark_analysis.png"
  plt.savefig(output_plot, dpi=300)
  plt.close("all")
  print(f"📈 Umfassender Analyse-Plot gespeichert unter: {output_plot}\n")


if __name__ == "__main__":
  analyze_and_compare()