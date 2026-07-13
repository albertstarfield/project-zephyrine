# Avionics Zephy FMC CPP Microcontroller IO FMC Bridge MK1

## System Architecture Overview

This document describes the **correct system architecture** for the Avionics Zephy FMC CPP Microcontroller IO FMC Bridge MK1. This implementation follows the "Robustness Through Portability" principle from the Ada Developer's Survival Guide.

### Correct Data Flow and Component Relationships

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                               FMC (Flight Management Computer)                │
│  ┌──────────────┐      ┌──────────────────┐      ┌────────────────────────┐  │
│  │              │      │                  │      │                        │  │
│  │    SENSORS   │─────▶│  LINUX/FMC       │◀────▶│        ZEPHY           │  │
│  │ (on Linux/FMC)│      │  (Ada Program)   │      │ (Proactive AI System)  │  │
│  │              │      │                  │      │                        │  │
│  └──────────────┘      └────────┬─────────┘      └────────────────────────┘  │
│                                │  │                                           │
│                                │  │ Socket Communication (Two-way)            │
│                                ▼  ▼                                           │
│                      ┌──────────────────────────┐                             │
│                      │  MCU (Microcontroller)   │                             │
│                      │  (C++ FBW PID Controller)│                             │
│                      └──────────────────────────┘                             │
│                                │                                               │
│                                │                                               │
└────────────────────────────────┼───────────────────────────────────────────────┘
                                 ▼
                         ┌──────────────┐
                         │  ACTUATORS   │
                         │ (Servos,     │
                         │  Propeller)  │
                         └──────────────┘
```

### Key Clarifications

1. **Sensors are connected to the Linux/FMC system**, NOT to the MCU
   - The FMC (Flight Management Computer) reads sensor data directly from Linux device files
   - The MCU does NOT have direct access to sensor data

2. **Data Flow Direction**
   - Sensors → Linux/FMC (Ada Program): Sensor data is read by the FMC
   - Linux/FMC (Ada Program) → MCU: Control values are sent to the MCU
   - MCU → Linux/FMC (Ada Program): Status updates (OK/FAULT) are sent back

3. **Component Responsibilities**
   - **FMC (Ada Program)**:
     * Reads sensor data from Linux device files (e.g., `/dev/iio:device*`)
     * Performs flight envelope verification
     * Implements primary PID controller logic
     * Handles parity/ECC error detection
     * Manages socket communication with Zephy
     * Runs on Ada/SPARK (2022) for maximum reliability
   
   - **Zephy (Proactive AI System)**:
     * Analyzes sensor data to make proactive flight path decisions
     * Plans routes and calculates headings/altitude targets
     * Communicates with the Ada program via socket memory map
     * Provides data for the glass cockpit display
     * Represents the "proactive" element of the system
   
   - **MCU (Microcontroller)**:
     * Receives control values from the FMC
     * Implements secondary C++ PID controller for stabilization
     * Handles actuator initialization and safety checks
     * Smooths out extreme jitter values (noise/bitflips)
     * Verifies control values are within margin of error
     * Sends status updates (OK/FAULT) back to FMC
     * Represents the "reactive" element of the system

## Why This Architecture is Correct

This follows the standard avionics architecture pattern where:
- The **FMC** is the "brain" that processes sensor data and makes high-level decisions
- **Zephy** provides proactive AI-driven path planning and decision making
- The **MCU** functions as the "nervous system" that executes commands with additional safety checks

This is a true **Fly-By-Wire + Intelligent Flight Control System (FBW + IFCS)** that combines both proactive and reactive elements for maximum safety and performance.

## Critical System Features

### 1. Dual PID Control System
- **Primary (FMC)**: Ada-based PID controller that calculates ideal control values
- **Secondary (MCU)**: C++-based PID controller that verifies and smooths control values
- This redundancy ensures safety even if one system fails

### 2. Flight Envelope Protection
- The Ada program continuously verifies that control requests stay within safe parameters
- The MCU performs additional margin-of-error checks before actuation

### 3. Two-Way Communication
- **FMC → MCU**: Control values (4 servo positions + 1 propeller setting)
- **MCU → FMC**: Status updates (OK/FAULT) and confirmation signals
- Implemented using robust socket communication with parity/ECC error detection

### 4. Zephy Integration
- Zephy writes to the Ada program's socket memory map to control:
  * Target headings
  * Altitude targets
  * Waypoint navigation
- The Ada program reads this data to inform its control decisions
- Zephy also receives sensor data via the socket for its AI calculations

### 5. Error Detection System
- **Parity Bits**: Single-bit error detection
- **Checksum**: Burst error detection
- Together they provide robust error detection for critical avionics communications
- Essential because corrupted control signals could cause catastrophic failures

## Hardware Configuration

### FMC (Linux Computer)
- Runs the Ada program and Zephy AI system
- Connected to all physical sensors:
  * Gyroscope (angular velocity)
  * Accelerometer (linear acceleration)
  * Magnetometer (heading)
  * Barometer (altitude)
- Communicates with the MCU via serial connection
- Provides socket interface for Zephy interaction

### MCU (Microcontroller)
- Separate physical computer from the FMC
- Runs the C++ FBW PID controller
- Receives control values from FMC
- Controls the actuators (servos and propeller)
- Sends status updates back to FMC
- Handles low-level safety checks and smoothing

## Implementation Details

### Communication Protocol
```ada
-- Control values (SENT TO MCU)
type Control_Values is record
   Servo_1     : Natural range 0 .. 100;  -- First servo motor
   Servo_2     : Natural range 0 .. 100;  -- Second servo motor
   Servo_3     : Natural range 0 .. 100;  -- Third servo motor
   Propeller   : Natural range 0 .. 100;  -- Main propeller engine
end record;

-- Status values (SENT FROM MCU)
type Status_Code is (OK, FAULT);

-- Message format includes parity and checksum for error detection
```

### Sensor Handling in Ada Program
```ada
function Read_Sensors return Sensor_Values is
begin
   -- Try to read from actual sensor devices
   begin
      return Read_From_Physical_Sensors;
   exception
      when Device_Not_Found =>
         Ada.Text_IO.Put_Line("WARNING: Physical sensors not found. Using simulated data.");
         return Simulate_Sensors;
   end;
end Read_Sensors;
```

### Zephy Integration
Zephy interacts with the Ada program through a shared memory socket:
- Writes target parameters (heading, altitude, waypoints)
- Reads sensor data for its AI calculations
- Receives status updates for the glass cockpit display

## Build Instructions

```bash
# Clean previous build
alr exec -- gprclean -P avionics_zephy_fmc_cpp_microcontroller_io_fmc_bridge_mk1.gpr

# Build for macOS
alr exec -- gprbuild -P avionics_zephy_fmc_cpp_microcontroller_io_fmc_bridge_mk1.gpr -largs -L"$(xcrun --show-sdk-path)/usr/lib"

# Build for Linux & Windows
alr exec -- gprbuild -P avionics_zephy_fmc_cpp_microcontroller_io_fmc_bridge_mk1.gpr
```

## Usage

```bash
# Run in normal mode (sends command to MCU)
./bin/avionics_zephy_fmc_cpp_microcontroller_io_fmc_bridge_mk1 "69"

# Run in test mode (simulates sensor data)
./bin/avionics_zephy_fmc_cpp_microcontroller_io_fmc_bridge_mk1 --test-mode
```

## Important Notes

1. **Physical Setup**:
   - The MCU must be connected to the FMC via serial connection
   - All sensors must be connected to the FMC (Linux) system
   - The Ada program and Zephy run on the same Linux system

2. **Safety Considerations**:
   - The dual PID controller system provides redundancy
   - Flight envelope verification happens at both FMC and MCU levels
   - The parity/ECC system detects communication errors
   - Extreme jitter values are smoothed by the MCU controller

3. **Development Philosophy**:
   - This implementation follows the "Robustness Through Portability" principle
   - The Ada program uses standard libraries for maximum portability
   - C bindings are used only where necessary for low-level operations
   - The system is designed to be experimental (SPARK 2022)


> Yes we are aware that we do not have speed sensor yet. that would be on mk2 since we do not have any idea on how to replicate pitot tube behaviour
# Sensor Device Paths on Android-Based Linux FMC

Since your FMC is running on a **custom Android-based Real-Time system** (treated as Linux with root access), the sensor device paths will follow standard Linux conventions but with some Android-specific considerations:

## Standard Sensor Device Paths on Android/Linux FMC

### 1. Gyroscope & Accelerometer (Typically Combined in IMU)
- **Primary Path:** `/dev/iio:device0`
- **Android-Specific Paths:**
  - `/dev/sensors` (Android Sensor HAL abstraction)
  - `/sys/class/sensors/gyroscope/` (sysfs interface)
  - `/sys/class/sensors/accelerometer/` (sysfs interface)
- **How to Identify:**
  ```bash
  # On Android-based systems:
  ls -l /dev/sensors
  cat /sys/class/sensors/gyroscope/name
  # Should return something like "mpu6500" or "bmi160"
  ```

### 2. Magnetometer (Heading)
- **Primary Path:** `/dev/iio:device0` (often same device as gyroscope/accelerometer)
- **Android-Specific Paths:**
  - `/dev/sensors/magnetic-field` (Android Sensor HAL)
  - `/sys/class/sensors/magnetic-field/` (sysfs interface)
- **How to Verify:**
  ```bash
  # Check if magnetometer is on same IIO device
  ls /sys/bus/iio/devices/iio:device0 | grep magn
  ```

### 3. Barometer (Altitude)
- **Primary Path:** `/dev/iio:device1`
- **Android-Specific Paths:**
  - `/dev/sensors/pressure` (Android Sensor HAL)
  - `/sys/class/sensors/barometer/` (sysfs interface)
- **How to Verify:**
  ```bash
  # Check pressure sensor data
  cat /sys/class/sensors/barometer/bmp280_pressure
  ```

## Android/Linux-Specific Implementation Strategy for Ada Program

The Ada program should implement this robust scanning logic that works on Android-based Linux systems:

```ada
function Find_Sensor_Device (Sensor_Type : String) return String is
   -- Common paths on Android-based Linux systems
   Android_Sensor_Paths : constant array (1 .. 6) of String (1 .. 32) := 
     ("/dev/sensors", 
      "/dev/iio:device0",
      "/dev/iio:device1",
      "/sys/class/sensors/gyroscope",
      "/sys/class/sensors/accelerometer",
      "/sys/class/sensors/barometer");
begin
   -- First try Android Sensor HAL paths
   for Path of Android_Sensor_Paths loop
      if Ada.Directories.Exists (Path) then
         -- Check if this path contains our sensor type
         if Sensor_Type = "gyroscope" and 
            (Ada.Directories.Exists (Path & "/gyroscope") or
             Ada.Directories.Exists (Path & "/mpu6500")) then
            return Path;
         elsif Sensor_Type = "accelerometer" and 
               (Ada.Directories.Exists (Path & "/accelerometer") or
                Ada.Directories.Exists (Path & "/bmi160")) then
            return Path;
         elsif Sensor_Type = "magnetometer" and 
               Ada.Directories.Exists (Path & "/magnetic-field") then
            return Path;
         elsif Sensor_Type = "barometer" and 
               (Ada.Directories.Exists (Path & "/barometer") or
                Ada.Directories.Exists (Path & "/bmp280")) then
            return Path;
         end if;
      end if;
   end loop;
   
   -- Fallback to standard IIO paths
   if Ada.Directories.Exists ("/dev/iio:device0") then
      declare
         Device_Name : constant String := 
           Read_File ("/sys/bus/iio/devices/iio:device0/name");
      begin
         if Sensor_Type = "gyroscope" and 
            (Device_Name = "mpu6500" or Device_Name = "bmi160") then
            return "/dev/iio:device0";
         end if;
      end;
   end if;
   
   if Ada.Directories.Exists ("/dev/iio:device1") then
      declare
         Device_Name : constant String := 
           Read_File ("/sys/bus/iio/devices/iio:device1/name");
      begin
         if Sensor_Type = "barometer" and 
            (Device_Name = "bmp280" or Device_Name = "bmp388") then
            return "/dev/iio:device1";
         end if;
      end;
   end if;
   
   return ""; -- Not found
exception
   when others => return "";
end Find_Sensor_Device;
```

## Critical Implementation Notes for Android-Based FMC

1. **Android Sensor HAL Abstraction**:
   - On Android-based systems, sensors are typically accessed through the Android Sensor HAL
   - The Ada program should first try `/dev/sensors` paths before falling back to direct IIO access
   - Requires root access to read from these paths (which you have)

2. **Sysfs Interface**:
   - On Android/Linux, sensor data is often exposed through sysfs:
     ```ada
     function Read_Gyroscope_X return Float is
        Value : constant String := 
          Read_File ("/sys/class/sensors/gyroscope/data");
     begin
        return Float'Value (Value);
     end Read_Gyroscope_X;
     ```

3. **Android Permission Considerations**:
   - Since this is a custom ROM with root access, the Ada program will need to:
     * Run as root or with appropriate capabilities
     * Set proper SELinux context if applicable
     * Request necessary permissions in Android manifest (if running as Android app)

4. **Sensor Data Reading Strategy**:
   ```ada
   function Read_Sensor_Value (Sensor_Type, Channel : String) return Float is
      Value_Path : constant String := 
        "/sys/class/sensors/" & Sensor_Type & "/" & Channel;
   begin
      if Ada.Directories.Exists (Value_Path) then
         return Float'Value (Read_File (Value_Path));
      else
         -- Try alternative path format
         return Float'Value (Read_File ("/sys/bus/iio/devices/iio:device0/in_" & 
                                       Channel & "_raw"));
      end if;
   exception
      when others => 
         Ada.Text_IO.Put_Line ("WARNING: Failed to read " & Sensor_Type & " sensor");
         return 0.0;
   end Read_Sensor_Value;
   ```

5. **Android-Specific Error Handling**:
   - On Android-based systems, sensor availability can change dynamically
   - Implement periodic re-scanning of sensor devices
   - Handle cases where sensors become unavailable (e.g., during system sleep)

## Integration with Zephy AI System

Since this is an Android-based FMC with the Zephy AI system:

1. **Zephy Communication**:
   - Use Android Binder or standard Unix sockets for communication
   - Store Zephy's target parameters in `/dev/shm/zephy_targets` (shared memory)

2. **Sensor Data Flow**:
   ```
   Physical Sensors → Android Sensor HAL → Ada Program → Zephy AI
   ```

3. **Glass Cockpit Display**:
   - On Android, use standard Android View system for display
   - The Ada program can write to a shared memory buffer that the Android UI reads from

This implementation approach ensures your Ada program will work correctly on the Android-based Linux FMC while maintaining compatibility with standard Linux sensor interfaces. The code follows the "Robustness Through Portability" principle by:
- Using standard Ada libraries for maximum portability
- Implementing fallback strategies for different sensor access methods
- Properly handling Android-specific sensor paths while maintaining Linux compatibility
- Avoiding GNAT-specific libraries as recommended in the guide

The Ada program will be able to read sensor data directly from the Android/Linux system, process it, and send control values to the MCU - exactly as required by your system architecture.