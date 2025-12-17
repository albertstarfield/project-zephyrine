
#include <cmath>
#include <iostream>

extern "C" {

    // --- CONSTANTS ---
    const double LAPSE_RATE = 0.0019812;      // deg C per ft
    const double STD_TEMP_C = 15.0;           // Sea level temp
    const double STD_TEMP_K = 288.15;         // Sea level temp Kelvin
    const double STD_PRESS_HG = 29.92;        // Sea level press
    const double ZERO_K = 273.15;             // 0C in Kelvin

    // ==========================================
    // MODULE 1: ALTITUDE
    // ==========================================

    double calc_pressure_alt(double indicated_alt, double altimeter_setting) {
        return (STD_PRESS_HG - altimeter_setting) * 1000.0 + indicated_alt;
    }

    double calc_std_temp_c(double altitude) {
        return STD_TEMP_C - (altitude * LAPSE_RATE);
    }

    double calc_density_alt(double pressure_alt, double oat_c) {
        double isa_temp = calc_std_temp_c(pressure_alt);
        return pressure_alt + (118.8 * (oat_c - isa_temp));
    }

    double calc_cloud_base(double oat_c, double dewpoint_c) {
        double spread = oat_c - dewpoint_c;
        return (spread / 2.5) * 1000.0;
    }

    // ==========================================
    // MODULE 2: AIRSPEED (NEW)
    // ==========================================
    
    // Helper: Calculate standard pressure ratio at altitude
    double get_pressure_ratio(double altitude) {
        // Standard atmospheric model formula
        double base = 1.0 - (0.0000068756 * altitude);
        return pow(base, 5.2561);
    }

    // Helper: Speed of Sound in Knots based on Temp (Celsius)
    // Formula: 38.9678 * sqrt(Temp_Kelvin)
    double calc_speed_of_sound_kts(double temp_c) {
        return 38.9678 * sqrt(temp_c + ZERO_K);
    }

    // TRUE AIRSPEED (TAS)
    // Inputs: Calibrated Airspeed (CAS), Pressure Alt (ft), OAT (C)
    // Uses Density Ratio method (sigma)
    double calc_tas(double cas, double p_alt, double oat_c) {
        
        // 1. Determine Density Ratio (Sigma)
        double p_ratio = get_pressure_ratio(p_alt);
        double t_ratio = (oat_c + ZERO_K) / STD_TEMP_K;
        double sigma = p_ratio / t_ratio;

        // 2. TAS Formula (Incompressible flow approx for <200kts)
        // TAS = CAS * (1 / sqrt(sigma))
        return cas * (1.0 / sqrt(sigma));
    }

    // MACH NUMBER
    // Inputs: TAS (knots), OAT (C)
    double calc_mach(double tas, double oat_c) {
        double speed_sound = calc_speed_of_sound_kts(oat_c);
        return tas / speed_sound;
    }

    // Health Check
    int flight_computer_ping() { return 888; }
}
