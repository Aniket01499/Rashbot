import subprocess
import struct
import os

class GameMemory:
    def __init__(self, process_name="RoadRash.exe"):
        self.process_name = process_name
        self.pid = self._get_pid()
        self.base_address = self._get_base_address()
        
        if not self.pid or not self.base_address:
            print(f"CRITICAL: Could not attach to {self.process_name}. Is it running?")
        else:
            print(f"Memory Hook Attached! PID: {self.pid} | Base: {hex(self.base_address)}")

    def _get_pid(self):
        """Finds the Linux PID of the Wine game."""
        try:
            pid_str = subprocess.check_output(["pgrep", "-f", self.process_name]).decode().strip().split('\n')[0]
            return int(pid_str)
        except subprocess.CalledProcessError:
            return None

    def _get_base_address(self):
        """Finds where the game is loaded in RAM (bypasses memory randomization)."""
        if not self.pid: return None
        try:
            with open(f"/proc/{self.pid}/maps", "r") as maps_file:
                for line in maps_file:
                    # Look for the main executable module
                    if self.process_name.lower() in line.lower():
                        # The first string is the memory range, e.g., '00400000-004a6000'
                        addr_range = line.split(" ")[0]
                        base_addr_str = addr_range.split("-")[0]
                        return int(base_addr_str, 16)
        except Exception as e:
            print(f"Failed to read memory maps: {e}")
        return None

    def read_int(self, offset):
        """Reads the exact value using the dynamic Base Address + Static Offset."""
        if not self.pid or not self.base_address:
            return None
            
        target_address = self.base_address + offset
            
        try:
            with open(f"/proc/{self.pid}/mem", "rb") as mem_file:
                mem_file.seek(target_address)
                data = mem_file.read(4) 
                return struct.unpack("<i", data)[0]
                
        except PermissionError:
            print("PERMISSION ERROR: You must run this script with 'sudo'!")
            return None
        except Exception:
            return None
