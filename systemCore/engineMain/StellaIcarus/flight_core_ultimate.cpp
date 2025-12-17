
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
