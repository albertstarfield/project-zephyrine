"""
STELLA ICARUS TRICKSHOT: CX-3 FLIGHT COMPUTER (ULTIMATE)
========================================================
Implements the complete physics suite of the ASA CX-3 Manual.
Target: High-performance C++ O3 Native Execution.

MODULES:
1. ATMOSPHERE (Altitudes, Cloud Base)
2. AIRSPEED (TAS, Mach)
3. WIND (Components, Triangle Solution)
4. PERFORMANCE (Glide, Climb, Fuel) [EXPANDED]
5. NAVIGATION (Compass, ETE) [NEW]
6. WEIGHT & BALANCE (Shift, %MAC) [EXPANDED]
"""

import ctypes
import os
import platform
import re
import subprocess
import sys
import time
from typing import Optional, Match

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
LIB_EXT = ".dll" if platform.system() == "Windows" else ".so"
LIB_NAME = f"flight_core_ultimate{LIB_EXT}"
SOURCE_FILE = "flight_core_ultimate.cpp"

# ==============================================================================
# 2. THE C++ ENGINE (ALL MODULES)
# ==============================================================================
cpp_source_code = """
#include <cmath>
#include <iostream>

extern "C" {

    // --- CONSTANTS ---
    const double NM_TO_FT = 6076.12; 
    const double STD_PRESS_HG = 29.92;
    const double STD_TEMP_C = 15.0;
    const double STD_TEMP_K = 288.15;
    const double ZERO_K = 273.15;
    const double PI = 3.14159265358979323846;

    // --- HELPERS ---
    double to_rad(double deg) { return deg * PI / 180.0; }
    double to_deg(double rad) { return rad * 180.0 / PI; }
    double get_pressure_ratio(double altitude) {
        return pow(1.0 - (0.0000068756 * altitude), 5.2561);
    }
    double calc_speed_of_sound_kts(double temp_c) {
        return 38.9678 * sqrt(temp_c + ZERO_K);
    }

    // ==========================================
    // 1. ALTITUDE & ATMOSPHERE
    // ==========================================
    double calc_pressure_alt(double indicated, double setting) {
        return (STD_PRESS_HG - setting) * 1000.0 + indicated;
    }
    double calc_density_alt(double p_alt, double oat_c) {
        double isa_temp = STD_TEMP_C - (p_alt * 0.0019812);
        return p_alt + (118.8 * (oat_c - isa_temp));
    }
    double calc_cloud_base(double temp, double dew) {
        return ((temp - dew) / 2.5) * 1000.0;
    }

    // ==========================================
    // 2. AIRSPEED
    // ==========================================
    double calc_tas(double cas, double p_alt, double oat_c) {
        double p_ratio = get_pressure_ratio(p_alt);
        double t_ratio = (oat_c + ZERO_K) / STD_TEMP_K;
        double sigma = p_ratio / t_ratio;
        return cas * (1.0 / sqrt(sigma));
    }
    double calc_mach(double tas, double oat_c) {
        return tas / calc_speed_of_sound_kts(oat_c);
    }

    // ==========================================
    // 3. WIND PHYSICS
    // ==========================================
    double calc_headwind(double speed, double dir, double runway) {
        return speed * cos(to_rad(dir - runway));
    }
    double calc_crosswind(double speed, double dir, double runway) {
        return speed * sin(to_rad(dir - runway));
    }
    double calc_groundspeed(double tas, double course, double wspd, double wdir) {
        double wca_rad = asin((wspd * sin(to_rad(wdir - course))) / tas); # Approx
        // Full vector math is safer for GS:
        // GS = TAS * cos(WCA) - HeadwindComponent
        // Let's use simple law of cosines vector addition for generic case
        double wind_angle = to_rad(wdir - course - 180); // Wind vector opposes flight
        double tas_x = 0; double tas_y = tas; // Relative to course
        double wind_x = wspd * sin(to_rad(wdir - course));
        double wind_y = wspd * cos(to_rad(wdir - course));
        // Simplified E6B output:
        double wca = to_deg(asin( (wspd * sin(to_rad(wdir-course))) / tas )); # Wind from right = +WCA
        double gs = (tas * cos(to_rad(wca))) - (wspd * cos(to_rad(wdir-course))); 
        return gs;
    }
    double calc_wca(double tas, double course, double wspd, double wdir) {
        // Wind From Right (+ value) requires heading change to right (+)
        return to_deg(asin( (wspd * sin(to_rad(wdir - course))) / tas ));
    }

    // ==========================================
    // 4. PERFORMANCE (GLIDE, CLIMB, FUEL)
    // ==========================================
    // GLIDE RATIO: Distance (NM) / Alt Lost (ft) -> Returns Ratio X:1
    double calc_glide_ratio(double dist_nm, double alt_lost_ft) {
        double dist_ft = dist_nm * NM_TO_FT;
        return dist_ft / alt_lost_ft;
    }
    // GLIDE DIST: Ratio, Alt Lost (ft) -> Returns NM
    double calc_glide_dist(double ratio, double alt_lost_ft) {
        double dist_ft = ratio * alt_lost_ft;
        return dist_ft / NM_TO_FT;
    }
    double calc_fuel_burn(double rate, double time) { return rate * time; }
    
    // CLIMB GRADIENT (Ft per NM)
    // Input: Ground Speed (Kts), Rate of Climb (FPM)
    // Formula: FPNM = ROC * 60 / GS
    double calc_climb_gradient(double gs, double roc) {
        if (gs == 0) return 0.0;
        return (roc * 60.0) / gs;
    }

    // ==========================================
    // 5. NAVIGATION
    // ==========================================
    // COMPASS HEADING
    // True Hdg +/- Var = Mag Hdg +/- Dev = Compass
    // Convention: West Var is (+), East Var is (-). 
    // "East is Least (Subtract), West is Best (Add)" applied to True to get Mag.
    // Wait, standard navigation: Mag = True - Variation(East). Mag = True + Variation(West).
    // User inputs signed numbers (e.g. -12 for 12E, +4 for 4W).
    double calc_compass(double true_hdg, double variation, double deviation) {
        return true_hdg + variation + deviation; // Assuming inputs are signed correctly
    }
    
    // ETE: Dist / Speed
    double calc_ete(double dist, double speed) {
        if (speed == 0) return 0.0;
        return dist / speed;
    }

    // ==========================================
    // 6. WEIGHT & BALANCE
    // ==========================================
    double calc_weight_shift(double total_wt, double d_cg, double d_dist) {
        if (d_dist == 0) return 0.0;
        return (total_wt * d_cg) / d_dist;
    }
    
    // %MAC (Mean Aerodynamic Chord)
    // Formula: %MAC = ((CG - LEMAC) / MAC_Length) * 100
    // Inputs: CG Location, LEMAC (Leading Edge MAC), MAC Length
    double calc_mac_percent(double cg, double lemac, double mac_len) {
        if (mac_len == 0) return 0.0;
        return ((cg - lemac) / mac_len) * 100.0;
    }

    int flight_computer_ping() { return 1000; }
}
"""

# ==============================================================================
# 3. DYNAMIC COMPILATION LOADER
# ==============================================================================
_LIB_HANDLE = None
_IS_OPTIMIZED = False

def _compile_and_load():
    global _LIB_HANDLE, _IS_OPTIMIZED
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, LIB_NAME)
    src_path = os.path.join(script_dir, SOURCE_FILE)

    if not os.path.exists(lib_path):
        try:
            print(f"[*] Flight Computer: Compiling Ultimate Core...", file=sys.stderr)
            with open(src_path, "w") as f:
                f.write(cpp_source_code)
            cmd = ["g++", "-O3", "-shared", "-fPIC", "-march=native", src_path, "-o", lib_path]
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[-] Compile failed: {e}", file=sys.stderr)
            return

    try:
        lib = ctypes.CDLL(lib_path)
        # Helper for applying types
        def set_sig(name, n_args):
            func = getattr(lib, name)
            func.argtypes = [ctypes.c_double] * n_args
            func.restype = ctypes.c_double

        # 2 Args
        for f in ['calc_pressure_alt', 'calc_density_alt', 'calc_cloud_base', 
                  'calc_mach', 'calc_fuel_burn', 'calc_glide_ratio', 
                  'calc_glide_dist', 'calc_climb_gradient', 'calc_ete']:
            set_sig(f, 2)
            
        # 3 Args
        for f in ['calc_tas', 'calc_headwind', 'calc_crosswind', 
                  'calc_weight_shift', 'calc_compass', 'calc_mac_percent']:
            set_sig(f, 3)

        # 4 Args
        for f in ['calc_wca', 'calc_groundspeed']:
            set_sig(f, 4)

        if lib.flight_computer_ping() == 1000:
            _LIB_HANDLE = lib
            _IS_OPTIMIZED = True
    except Exception as e:
        print(f"[-] Load failed: {e}", file=sys.stderr)

_compile_and_load()

# ==============================================================================
# 4. REGEX TRIGGERS (ULTIMATE)
# ==============================================================================
PATTERN = re.compile(
    r"(?is).*" 
    r"(?:" 
        r"(?P<help_req>\bhelp\s+(?:flight\s+computer|aviation\s+calc)\b)" 
        r"|"
        r"(?:\b(?:calc|calculate|find|solve)\b.*?"
           # Huge list of captured commands
           r"(?P<type>density altitude|pressure altitude|cloud base|"
                   r"true airspeed|mach number|"
                   r"wind component|wind solution|"
                   r"fuel burn|glide ratio|glide distance|climb gradient|ete|"
                   r"compass heading|mac percent|weight shift)\b.*?"
           r"(?P<val1>-?[\d\.]+).*?"
           r"(?P<val2>-?[\d\.]+)"
           r"(?:.*?(?P<val3>-?[\d\.]+))?"
           r"(?:.*?(?P<val4>-?[\d\.]+))?"
        r")"
    r")" 
    r".*"
)

# ==============================================================================
# 5. HANDLER
# ==============================================================================
def handler(match: Match[str], user_input: str, session_id: str) -> Optional[str]:
    if not _IS_OPTIMIZED:
        return "Flight Computer Core is offline."

    # --- HELP ---
    if match.group("help_req"):
        return (
            "**TRICKSHOT ULTIMATE FLIGHT COMPUTER**\n"
            "Supported Functions:\n"
            "- **Alt:** `pressure altitude`, `density altitude`, `cloud base`\n"
            "- **Spd:** `true airspeed`, `mach number`\n"
            "- **Wind:** `wind component`, `wind solution` (GS/WCA)\n"
            "- **Perf:** `fuel burn`, `glide ratio`, `glide distance`, `climb gradient`\n"
            "- **Nav:** `ete`, `compass heading`\n"
            "- **W/B:** `weight shift`, `mac percent`\n\n"
            "Example: `calculate glide distance 12 5000` (Ratio 12:1, 5000ft drop)"
        )

    # --- PARSING ---
    t = match.group("type").lower()
    try:
        v1 = float(match.group("val1"))
        v2 = float(match.group("val2"))
        v3 = float(match.group("val3")) if match.group("val3") else 0.0
        v4 = float(match.group("val4")) if match.group("val4") else 0.0
    except: return "Input Error: Numeric values required."

    start = time.perf_counter()
    out = ""

    # --- LOGIC ROUTING ---
    if "pressure altitude" in t:
        out = f"Pressure Alt: {_LIB_HANDLE.calc_pressure_alt(v1, v2):.0f} ft"
    elif "density altitude" in t:
        out = f"Density Alt: {_LIB_HANDLE.calc_density_alt(v1, v2):.0f} ft"
    elif "cloud base" in t:
        out = f"Cloud Base: {_LIB_HANDLE.calc_cloud_base(v1, v2):.0f} ft AGL"
    elif "true airspeed" in t:
        if not match.group("val3"): return "Need 3 inputs: CAS, Alt, Temp"
        res = _LIB_HANDLE.calc_tas(v1, v2, v3)
        out = f"TAS: {res:.1f} kts (Mach {_LIB_HANDLE.calc_mach(res, v3):.3f})"
    elif "mach number" in t:
        out = f"Mach: M{_LIB_HANDLE.calc_mach(v1, v2):.3f}"
    elif "wind component" in t:
        if not match.group("val3"): return "Need 3 inputs: WindSpd, WindDir, Rwy"
        hw = _LIB_HANDLE.calc_headwind(v1, v2, v3)
        xw = _LIB_HANDLE.calc_crosswind(v1, v2, v3)
        out = f"Headwind: {hw:.1f} kts | Crosswind: {xw:.1f} kts"
    elif "wind solution" in t:
        if not match.group("val4"): return "Need 4 inputs: TAS, Crs, WSpd, WDir"
        gs = _LIB_HANDLE.calc_groundspeed(v1, v2, v3, v4)
        wca = _LIB_HANDLE.calc_wca(v1, v2, v3, v4)
        out = f"GroundSpeed: {gs:.1f} kts | WCA: {wca:.1f}° | Hdg: {v2+wca:.0f}°"
    elif "fuel burn" in t:
        out = f"Fuel Burn: {_LIB_HANDLE.calc_fuel_burn(v1, v2):.1f} units"
    
    # --- NEW ULTIMATE FUNCTIONS ---
    elif "glide ratio" in t:
        # v1=Dist(NM), v2=Alt(Ft)
        out = f"Glide Ratio: {_LIB_HANDLE.calc_glide_ratio(v1, v2):.1f} : 1"
    elif "glide distance" in t:
        # v1=Ratio, v2=Alt(Ft)
        out = f"Glide Distance: {_LIB_HANDLE.calc_glide_dist(v1, v2):.1f} NM"
    elif "climb gradient" in t:
        # v1=GS, v2=ROC
        out = f"Climb Gradient: {_LIB_HANDLE.calc_climb_gradient(v1, v2):.0f} Ft/NM"
    elif "ete" in t:
        # v1=Dist, v2=GS
        out = f"ETE: {_LIB_HANDLE.calc_ete(v1, v2):.2f} hours"
    elif "compass heading" in t:
        # v1=TrueHdg, v2=Var, v3=Dev
        if not match.group("val3"): return "Need 3 inputs: TrueHdg, Var(+W/-E), Dev"
        res = _LIB_HANDLE.calc_compass(v1, v2, v3)
        out = f"Compass Heading: {res:.0f}°"
    elif "mac percent" in t:
        # v1=CG, v2=LEMAC, v3=MAC_Len
        if not match.group("val3"): return "Need 3 inputs: CG, LEMAC, MAC_Len"
        out = f"%MAC: {_LIB_HANDLE.calc_mac_percent(v1, v2, v3):.1f}%"
    elif "weight shift" in t:
        if not match.group("val3"): return "Need 3 inputs: TotalWt, DeltaCG, Dist"
        out = f"Shift Weight: {_LIB_HANDLE.calc_weight_shift(v1, v2, v3):.1f} lbs"

    lat = (time.perf_counter() - start) * 1e9
    return f"**{out}**\n[Trickshot: {lat:.2f} ns]"

# ==============================================================================
# 6. DEV TEST
# ==============================================================================
if __name__ == "__main__":
    tests = [
        "calculate glide distance 12 2000",   # Glide ratio 12:1, lost 2000ft (Page 13)
        "calculate climb gradient 120 800",   # 120kts GS, 800fpm climb
        "calculate compass heading 350 12 -2", # True 350, 12 West Var, -2 Dev
        "calculate mac percent 400 350 150"   # CG 400, LEMAC 350, MAC 150
    ]
    print("--- TRICKSHOT ULTIMATE ---")
    for t in tests:
        m = PATTERN.match(t)
        if m:
            print(f"In: {t}\nOut: {handler(m, t, 'test')}\n")