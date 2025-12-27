import time
import psutil
import platform
import subprocess
import ctypes
import sys

def get_idle_duration_debug():
    system = platform.system()
    try:
        if system == 'Darwin': # macOS
            # Command to get idle time in nanoseconds
            cmd = "ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF; exit}'"
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
            raw_out = result.stdout.strip()
            print(f"  [Raw ioreg output]: '{raw_out}'")
            if raw_out:
                return int(raw_out) / 1_000_000_000
        elif system == 'Windows':
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
                return millis / 1000.0
        elif system == 'Linux':
            try:
                result = subprocess.run(["xprintidle"], stdout=subprocess.PIPE, text=True)
                return float(result.stdout.strip()) / 1000.0
            except FileNotFoundError:
                print("  [Error]: xprintidle not found")
                return 0.0
    except Exception as e:
        print(f"  [Exception]: {e}")
        return 0.0
    return 0.0

def debug_loop():
    print(f"--- System Monitor Debug (OS: {platform.system()}) ---")
    print("Press Ctrl+C to stop.")
    
    while True:
        # 1. Idle Time
        idle_sec = get_idle_duration_debug()
        
        # 2. Power
        try:
            battery = psutil.sensors_battery()
            plugged = battery.power_plugged if battery else True
            percent = battery.percent if battery else 100
        except:
            plugged = True
            percent = "N/A"

        # 3. Resources
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent
        
        # 4. Logic Simulation (Mode -5)
        # Rule: Idle > 300s AND Plugged In AND CPU < 70% AND RAM < 80% -> RELEASE (0.0)
        # Else -> HALT (0.98)
        
        decision = "HALT (0.98)" # Default state
        
        fail_reasons = []
        if idle_sec < 300: fail_reasons.append(f"Active (Idle {idle_sec:.1f}s < 300s)")
        if not plugged: fail_reasons.append("On Battery")
        if cpu >= 70.0: fail_reasons.append(f"CPU High ({cpu}%)")
        if ram >= 80.0: fail_reasons.append(f"RAM High ({ram}%)")
        
        if not fail_reasons:
            decision = "RELEASE (0.0)"
        
        # Output
        print("-" * 60)
        print(f"TIME      : {time.strftime('%H:%M:%S')}")
        print(f"IDLE      : {idle_sec:.2f} seconds")
        print(f"POWER     : {'AC/Plugged' if plugged else 'Battery'} ({percent}%)")
        print(f"RESOURCES : CPU: {cpu}% | RAM: {ram}%")
        print(f"LOGIC -5  : {decision}")
        if fail_reasons:
            print(f"BLOCKERS  : {', '.join(fail_reasons)}")
            
        time.sleep(1.0)

if __name__ == "__main__":
    debug_loop()
