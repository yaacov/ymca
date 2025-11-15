"""
Llama.cpp Integration - Convert models to GGUF format using llama.cpp.
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LlamaCppConverter:
    """Handle GGUF conversion using llama.cpp."""
    
    def __init__(self):
        """Initialize the llama.cpp converter."""
        self.llama_cpp_path = self._find_llama_cpp()
    
    def convert(
        self,
        model_path: Path,
        quantization: Optional[str] = None,
        output_type: str = "f16"
    ) -> Path:
        """Convert a model to GGUF format.
        
        Args:
            model_path: Path to the model directory (safetensor format)
            quantization: Quantization method (e.g., 'q4_k_m', 'q5_k_m', 'q6_k', 'q8_0')
                         K-quants (q4_k_m, q5_k_m, etc.) recommended for better quality
            output_type: Output data type (f32, f16, q8_0, etc.)
            
        Returns:
            Path to the converted GGUF file
        """
        logger.info("Converting model to GGUF format...")
        
        # Determine output paths
        model_name = model_path.parent.name
        gguf_dir = model_path.parent / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        
        base_output = gguf_dir / f"{model_name}-{output_type}.gguf"
        final_output = gguf_dir / f"{model_name}-{quantization}.gguf" if quantization else base_output
        
        logger.info(f"Input: {model_path}")
        logger.info(f"Output: {final_output}")
        
        try:
            if self.llama_cpp_path:
                logger.info(f"Found llama.cpp at: {self.llama_cpp_path}")
                
                # Convert to GGUF
                self._run_conversion(model_path, base_output, output_type)
                
                # Quantize if requested
                if quantization:
                    self._run_quantization(base_output, final_output, quantization)
                    # Remove intermediate file
                    if base_output.exists() and base_output != final_output:
                        base_output.unlink()
                        logger.info(f"Removed intermediate file: {base_output}")
                
                logger.info(f"✓ GGUF conversion completed: {final_output}")
                return final_output
            else:
                logger.warning("llama.cpp not found. Please install it to enable conversion.")
                return self._create_instructions(model_path, final_output, output_type, quantization, gguf_dir)
                
        except Exception as e:
            logger.error(f"Failed to convert model to GGUF: {e}")
            raise
    
    def _find_llama_cpp(self) -> Optional[Path]:
        """Find llama.cpp installation in common locations."""
        search_paths = [
            Path("llama.cpp"),
            Path("../llama.cpp"),
            Path.home() / "llama.cpp",
            Path("/usr/local/llama.cpp"),
        ]
        
        for path in search_paths:
            if path.exists():
                # Check for convert scripts (both dash and underscore versions)
                if any([
                    (path / "convert_hf_to_gguf.py").exists(),
                    (path / "convert-hf-to-gguf.py").exists(),
                    (path / "examples" / "convert-hf-to-gguf.py").exists(),
                    (path / "convert.py").exists()
                ]):
                    return path.resolve()
        
        # Check PATH
        if shutil.which("convert_hf_to_gguf.py") or shutil.which("convert-hf-to-gguf.py"):
            script = shutil.which("convert_hf_to_gguf.py") or shutil.which("convert-hf-to-gguf.py")
            return Path(script).parent
        
        return None
    
    def _run_conversion(self, model_path: Path, output_file: Path, output_type: str):
        """Run llama.cpp conversion script."""
        if not self.llama_cpp_path:
            raise RuntimeError("llama.cpp not found")
        
        # Find convert script
        convert_script = self._find_convert_script()
        
        logger.info(f"Running conversion script: {convert_script}")
        
        cmd = [
            sys.executable,
            str(convert_script),
            str(model_path),
            "--outtype", output_type,
            "--outfile", str(output_file)
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.llama_cpp_path))
        
        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
        
        logger.info("✓ Model converted to GGUF format")
    
    def _run_quantization(self, input_file: Path, output_file: Path, quantization: str):
        """Run llama.cpp quantization tool."""
        if not self.llama_cpp_path:
            raise RuntimeError("llama.cpp not found")
        
        quantize_bin = self._find_quantize_binary()
        
        logger.info(f"Using quantize binary: {quantize_bin}")
        logger.info(f"Quantizing model with method: {quantization}")
        
        cmd = [str(quantize_bin), str(input_file), str(output_file), quantization]
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Quantization failed: {result.stderr}")
            raise RuntimeError(f"Quantization failed: {result.stderr}")
        
        logger.info(f"✓ Model quantized with {quantization}")
    
    def _find_convert_script(self) -> Path:
        """Find the conversion script in llama.cpp directory."""
        if not self.llama_cpp_path:
            raise RuntimeError("llama.cpp not found")
        
        candidates = [
            self.llama_cpp_path / "convert_hf_to_gguf.py",        # Underscore version (current)
            self.llama_cpp_path / "convert-hf-to-gguf.py",        # Dash version (legacy)
            self.llama_cpp_path / "examples" / "convert-hf-to-gguf.py",
            self.llama_cpp_path / "convert.py"
        ]
        
        for script in candidates:
            if script.exists():
                return script
        
        raise FileNotFoundError(f"Convert script not found in {self.llama_cpp_path}")
    
    def _find_quantize_binary(self) -> Path:
        """Find the quantize binary, building if necessary."""
        if not self.llama_cpp_path:
            raise RuntimeError("llama.cpp not found")
        
        locations = [
            self.llama_cpp_path / "build" / "bin" / "llama-quantize",  # CMake current
            self.llama_cpp_path / "build" / "tools" / "quantize",  # CMake tools
            self.llama_cpp_path / "build" / "bin" / "quantize",  # CMake legacy
            self.llama_cpp_path / "build" / "bin" / "Release" / "quantize",  # CMake Windows
            self.llama_cpp_path / "build" / "quantize",
            self.llama_cpp_path / "quantize",  # Legacy make
            self.llama_cpp_path / "llama-quantize",  # Legacy make (new name)
        ]
        
        for location in locations:
            if location.exists():
                return location
        
        # Try to build
        logger.info("Quantize binary not found, building with CMake...")
        self._build_llama_cpp()
        
        # Check again
        for location in locations:
            if location.exists():
                return location
        
        raise RuntimeError("Quantize binary not found after build")
    
    def _build_llama_cpp(self):
        """Build llama.cpp using CMake."""
        if not self.llama_cpp_path:
            raise RuntimeError("llama.cpp not found")
        
        build_dir = self.llama_cpp_path / "build"
        build_dir.mkdir(exist_ok=True)
        
        # Configure
        result = subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
            capture_output=True,
            text=True,
            cwd=str(build_dir)
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"CMake configure failed: {result.stderr}")
        
        # Build
        result = subprocess.run(
            ["cmake", "--build", ".", "--config", "Release"],
            capture_output=True,
            text=True,
            cwd=str(build_dir)
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"CMake build failed: {result.stderr}")
        
        logger.info("✓ Built llama.cpp")
    
    def _create_instructions(
        self,
        model_path: Path,
        output_file: Path,
        output_type: str,
        quantization: Optional[str],
        gguf_dir: Path
    ) -> Path:
        """Create instruction file for manual conversion."""
        instruction_file = gguf_dir / "conversion_instructions.txt"
        
        with open(instruction_file, 'w') as f:
            f.write("GGUF Conversion Instructions\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model downloaded to: {model_path}\n\n")
            f.write("To convert this model to GGUF format:\n\n")
            f.write("1. Install llama.cpp:\n")
            f.write("   git clone https://github.com/ggerganov/llama.cpp\n")
            f.write("   cd llama.cpp && mkdir -p build && cd build\n")
            f.write("   cmake .. && cmake --build . --config Release\n\n")
            f.write("2. Convert to GGUF:\n")
            f.write(f"   python llama.cpp/convert-hf-to-gguf.py {model_path} \\\n")
            f.write(f"     --outtype {output_type} --outfile {output_file}\n\n")
            
            if quantization:
                f.write("3. Quantize:\n")
                f.write(f"   ./llama.cpp/build/bin/quantize {output_file} \\\n")
                f.write(f"     {output_file.with_stem(output_file.stem + '-' + quantization)} {quantization}\n\n")
            
            f.write("\nFor more information:\n")
            f.write("https://github.com/ggerganov/llama.cpp\n")
        
        logger.info(f"Conversion instructions saved to: {instruction_file}")
        return instruction_file

