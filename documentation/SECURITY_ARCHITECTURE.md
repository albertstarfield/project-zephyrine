# Hardware-Bound Key Derivation Architecture

## Core Principle

```
Integrity Hash (computed fresh each boot) + User Secret (password/recovery key) → Master Key (512-bit, memory only)
```

- No key file stored on disk
- Key exists ONLY in Ada runtime memory (SPARK-verified package)
- Process exit → key gone
- Cold boot attack required to extract

## Detection Method

- Detect hash mismatch via DECRYPTION FAILURE (not hash comparison)
- Store encrypted test blob in `system_state` table (SQLite)
- On boot, try decrypt test blob with derived key
- If fails → signal `run.py` via stdio → prompt user for password/recovery key

## Key Derivation Chain

```
1. Compute integrity_hash = SHA512(hw_hash || binary_hash) from system state
2. master_key (512-bit) = HKDF-SHA512(salt=integrity_hash, ikm=user_secret, info="adelaide:master-key:v1")
3. aes_key (256-bit) = HKDF-SHA256(salt=master_key, ikm=context_string, info="adelaide:db:memory:v1")
4. Use aes_key for AES-256-GCM field encryption
```

## Recovery Key Mechanism

- First boot: prompt user for password (like phone setup)
- User provides password → derives key → encrypts test blob → stores in `system_state`
- Subsequent boots: recompute integrity_hash → derive key → try decrypt test blob
- If hardware changed: integrity_hash changes → decrypt fails → prompt user again
- User can:
  - (a) enter same password (works if same hardware)
  - (b) enter recovery key
  - (c) generate new key (re-encrypts all data)

## Hardware Identity Sources

### Linux
| Component | Source |
|-----------|--------|
| USB devices | `lsusb` |
| System info | `lshw -c system` |
| PCI devices | `lspci` |
| BIOS/Serial | `dmidecode -t system` |
| CPU | `/proc/cpuinfo` |
| RAM | `dmidecode -t memory` |
| Disk serial | `lsblk -d -o NAME,SERIAL` |

### macOS
| Component | Source |
|-----------|--------|
| USB devices | `system_profiler SPUSBDataType` |
| Hardware info | `system_profiler SPHardwareDataType` |
| PCI devices | `system_profiler SPPCIDataType` |
| Hardware tree | `ioreg -l` (IOPlatformSerialNumber, IOPlatformUUID) |
| NVMe | `system_profiler SPNVMeDataType` |
| CPU | `sysctl machdep.cpu` |
| RAM | `system_profiler SPMemoryDataType` |
| Thunderbolt | `system_profiler SPThunderboltDataType` |

## Binary Integrity Sources

### Linux
| Component | Source |
|-----------|--------|
| Kernel | `/boot/*vmlinuz*`, `/boot/*initrd*` |
| Bootloader | `/boot/efi/*` |
| Core utils | `/bin/*`, `/usr/bin/*` |
| Systemd | `/etc/systemd/system/*` |

### macOS
| Component | Source |
|-----------|--------|
| Kernel | `/System/Library/Kernels/*` (SIP-protected, scan anyway) |
| Bootloader | `/System/Library/CoreServices/boot.efi` |
| Homebrew | `/usr/local/bin/*` |
| LaunchDaemons | `/Library/LaunchDaemons/*` |
| LaunchAgents | `/Library/LaunchAgents/*` |
| Kernel Extensions | `/Library/Extensions/*` |

**NOTE:** Even SIP-protected paths must be scanned. SIP has zero-day vulnerabilities.

## SPARK-Verified Key Storage (512-bit)

```ada
package Master_Key_Store
  with SPARK_Mode => On
is
   subtype Key_Index is Positive range 1 .. 64;  -- 512 bits = 64 bytes
   type Key_Type is array (Key_Index) of Interfaces.Unsigned_8
     with Pack;

   procedure Set_Key (K : Key_Type)
     with Global => null;

   function Get_Key return Key_Type
     with Global => null;

   procedure Clear_Key
     with Global => null;

   function Is_Set return Boolean
     with Global => null;

private
   Key       : Key_Type := (others => 0);
   Key_Valid : Boolean := False;
end Master_Key_Store;
```

## stdio Protocol

### Ada → run.py
- `INTEGRITY_MISMATCH` - key derivation failed
- `INVALID_SECRET` - user provided wrong password
- `KEY_ACCEPTED` - key verified successfully
- `READY` - startup complete

### run.py → Ada
- User secret (password or recovery key) followed by newline

## KISS Mode (Phone-like Setup)

### First boot
```
  Welcome to Adelaide.

  Let's set up your password.
  This password protects your data.
  You'll need it every time Adelaide starts.

  Create password: [input]
  Confirm password: [input]
  Password set.

  Your recovery key is: XXXX-XXXX-XXXX-XXXX
  WRITE THIS DOWN. It's your backup if you forget your password.

  Press Enter to continue...
```

### Subsequent boot
```
  Welcome back.
  Please enter your password: [input]
  Verifying... Access granted.
```

### Hardware change
```
  Hardware change detected.
  Please enter your password or recovery key: [input]
```

## Migration from Old Key System

1. Detect old key file at `config/master.key` (or legacy `~/.config/adelaide/master.key`)
2. Prompt user for password
3. Derive new master_key with hardware-bound integrity hash
4. Load old key from file
5. Re-encrypt all databases with new key
6. Delete old key file
7. Store integrity_test blob with new key

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/master_key_store.ads` | NEW - SPARK-verified key storage |
| `src/master_key_store.adb` | NEW - Key storage implementation |
| `src/system_integrity.ads` | NEW - Platform-adaptive hash computation |
| `src/system_integrity.adb` | NEW - Hash computation implementation |
| `src/key_derivation.ads` | NEW - HKDF-SHA512 key derivation |
| `src/key_derivation.adb` | NEW - Key derivation implementation |
| `src/adl_crypto.c` | MODIFY - Add HKDF-SHA512 functions |
| `src/adl_crypto.h` | MODIFY - Add SHA512 declarations |
| `src/database_manager.adb` | MODIFY - Add integrity test blob verification |
| `src/adelaide_server.adb` | MODIFY - Add stdio protocol for key exchange |
| `run.py` | MODIFY - Add stdio handler, KISS mode prompts |
| `adelaide_lite.gpr` | MODIFY - Add new source files |
