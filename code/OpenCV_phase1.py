import threading
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynvml
import psutil
import torch


class TelemetryMonitor:

  def __init__(self, interval=0.01):
    self.interval = interval
    self.stop_event = threading.Event()
    self.timestamps = []
    self.ram_history = []
    self.vram_history = []
    self.util_history = []
    self.temp_history = []
    self.thread = None

    try:
      pynvml.nvmlInit()
      self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      self.nvml_active = True
    except Exception:
      self.nvml_active = False

  def _loop(self):
    process = psutil.Process()
    start_time = time.time()
    while not self.stop_event.is_set():
      t = time.time() - start_time
      self.timestamps.append(t)
      self.ram_history.append(process.memory_info().rss / (1024 ** 2))

      if torch.cuda.is_available():
        self.vram_history.append(
            torch.cuda.memory_allocated() / (1024 ** 2)
        )
      else:
        self.vram_history.append(0)

      if self.nvml_active:
        try:
          util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
          self.util_history.append(util.gpu)
          temp = pynvml.nvmlDeviceGetTemperature(
              self.handle, pynvml.NVML_TEMPERATURE_GPU
          )
          self.temp_history.append(temp)
        except Exception:
          self.util_history.append(0)
          self.temp_history.append(0)
      else:
        self.util_history.append(0)
        self.temp_history.append(0)

      time.sleep(self.interval)

  def start(self):
    self.timestamps.clear()
    self.ram_history.clear()
    self.vram_history.clear()
    self.util_history.clear()
    self.temp_history.clear()
    self.stop_event.clear()
    self.thread = threading.Thread(target=self._loop, daemon=True)
    self.thread.start()

  def stop(self):
    self.stop_event.set()
    if self.thread:
      self.thread.join()
    if self.nvml_active:
      try:
        pynvml.nvmlShutdown()
      except Exception:
        pass

  def get_current_metrics(self):
    if self.ram_history:
      return {
          "RAM_MB": self.ram_history[-1],
          "VRAM_MB": self.vram_history[-1],
          "GPU_Load_Percent": self.util_history[-1],
          "GPU_Temp_C": self.temp_history[-1],
      }
    return {"RAM_MB": 0, "VRAM_MB": 0, "GPU_Load_Percent": 0, "GPU_Temp_C": 0}


def run_phase1_baseline(duration_seconds=30):
  print(
      f"🎥 Starte Phase 1 (Baseline): CPU Webcam-Capture (Ziel:"
      f" {duration_seconds} Sekunden)..."
  )

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  if not cap.isOpened():
    print("❌ Fehler: Webcam konnte nicht geöffnet werden!")
    return

  monitor = TelemetryMonitor(interval=0.01)
  monitor.start()

  frame_records = []
  start_time = time.time()
  last_frame_time = start_time
  frame_count = 0

  try:
    while (time.time() - start_time) < duration_seconds:
      ret, frame = cap.read()
      if not ret:
        break

      current_time = time.time()
      dt = current_time - last_frame_time
      instant_fps = 1.0 / dt if dt > 0 else 0.0
      last_frame_time = current_time

      frame_count += 1
      elapsed = current_time - start_time
      sys_metrics = monitor.get_current_metrics()

      frame_records.append({
          "Timestamp": elapsed,
          "Frame_Index": frame_count,
          "Instant_FPS": round(instant_fps, 2),
          "RAM_MB": round(sys_metrics["RAM_MB"], 2),
          "VRAM_MB": round(sys_metrics["VRAM_MB"], 2),
          "GPU_Load_Percent": sys_metrics["GPU_Load_Percent"],
          "GPU_Temp_C": sys_metrics["GPU_Temp_C"],
      })

      cv2.imshow("Phase 1 - Baseline CPU (Druecke 'q' zum Beenden)", frame)
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
  finally:
    cap.release()
    cv2.destroyAllWindows()
    monitor.stop()

  total_time = time.time() - start_time
  avg_fps = frame_count / total_time if total_time > 0 else 0

  df = pd.DataFrame(frame_records)
  csv_filename = "phase1_telemetry_detailed.csv"
  df.to_csv(csv_filename, index=False)

  print("\n" + "=" * 50)
  print("📊 DIREKT-AUSWERTUNG: PHASE 1 (BASELINE)")
  print("=" * 50)
  print(f"⏱️ Gesamtdauer:          {total_time:.2f} Sekunden")
  print(f"🎞️ Gelesene Frames:      {frame_count}")
  print(f"🚀 Durchschnittliche FPS: {avg_fps:.2f}")
  print(f"💾 Peak System-RAM:      {df['RAM_MB'].max():.2f} MB")
  print(f"🎮 Peak GPU-VRAM:        {df['VRAM_MB'].max():.2f} MB")
  print(f"🔥 Max. GPU-Temperatur:  {df['GPU_Temp_C'].max():.1f} °C")
  print(f"📁 Detaillierte CSV gespeichert unter: {csv_filename}")
  print("=" * 50)

  # Plot erstellen
  plt.figure(figsize=(12, 8), dpi=300)
  plt.subplot(3, 1, 1)
  plt.plot(
      df["Timestamp"],
      df["Instant_FPS"],
      color="purple",
      label="Instant FPS",
      alpha=0.7,
  )
  plt.axhline(avg_fps, color="red", linestyle="--", label=f"Ø FPS ({avg_fps:.1f})")
  plt.ylabel("FPS")
  plt.title(
      f"Phase 1: CPU Baseline Performance (Ø FPS: {avg_fps:.1f})",
      fontweight="bold",
  )
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend(loc="upper right")

  plt.subplot(3, 1, 2)
  plt.plot(
      df["Timestamp"],
      df["GPU_Load_Percent"],
      color="crimson",
      label="GPU Load (%)",
  )
  plt.plot(
      df["Timestamp"],
      df["GPU_Temp_C"],
      color="darkorange",
      label="GPU Temp (°C)",
  )
  plt.ylabel("Prozent / °C")
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend(loc="upper right")

  plt.subplot(3, 1, 3)
  plt.plot(df["Timestamp"], df["RAM_MB"], color="forestgreen", label="RAM (MB)")
  plt.plot(
      df["Timestamp"], df["VRAM_MB"], color="dodgerblue", label="VRAM (MB)"
  )
  plt.xlabel("Zeit in Sekunden")
  plt.ylabel("Speicher (MB)")
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend(loc="upper right")

  plt.tight_layout()
  plt.savefig("phase1_detailed_report.png", dpi=300)
  plt.close("all")
  print("📈 Report-Plot gespeichert unter: phase1_detailed_report.png\n")


if __name__ == "__main__":
  run_phase1_baseline(duration_seconds=30)