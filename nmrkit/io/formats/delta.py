import struct
from typing import Dict, Optional, List

import numpy as np

from nmrkit.core import NMRData, DimensionInfo, LinearGenerator
from nmrkit.io.base import FormatReader
from nmrkit.utils import complexify


class DeltaReader(FormatReader):
    # Maximum number of dimensions supported
    MAX_DIMENSIONS = 8

    # Header offsets (in bytes)
    OFFSET_ENDIAN_FLAG = 8  # Endianness flag
    OFFSET_DIM_COUNT = 12  # Number of dimensions
    OFFSET_DATA_TYPE = 14  # Data type byte
    OFFSET_DIM_TYPES = 24  # Dimension type flags
    OFFSET_DIM_SIZES = 176  # Dimension sizes
    OFFSET_AXIS_START = 272  # Axis start values
    OFFSET_AXIS_STOP = 336  # Axis stop values
    OFFSET_BASE_FREQ = 1064  # Base frequencies
    OFFSET_ZERO_POINT = 1128  # Zero points (offsets)
    OFFSET_PARAM_START = 1212  # Parameter start position
    OFFSET_PARAM_LENGTH = 1216  # Parameter length in bytes
    OFFSET_DATA_START = 1284  # Data start position
    OFFSET_DATA_LENGTH = 1288  # Data length in bytes

    # Dimension types
    DIM_TYPE_NONE = 0  # No type
    DIM_TYPE_REAL = 1  # Real data
    DIM_TYPE_TPPI = 2  # TPPI (Time-Proportional Phase Incrementation)
    DIM_TYPE_COMPLEX = 3  # Complex data
    DIM_TYPE_REAL_COMPLEX = 4  # Real-complex data (Magnitude)
    DIM_TYPE_ENVELOPE = 5  # Envelope

    # Bitmask for data type byte
    PRECISION_BIT = 0x40  # 7th bit for float32 precision

    # Block size for data reshaping
    BLOCK_SIZE = 32

    # Endian modes
    BIG_ENDIAN = 0  # Big-endian byte order
    LITTLE_ENDIAN = 1  # Little-endian byte order

    # Data formats
    FORMAT_1D = 1  # 1D NMR data
    FORMAT_2D = 2  # 2D NMR data
    FORMAT_3D = 3  # 3D NMR data
    FORMAT_4D = 4  # 4D NMR data
    FORMAT_5D = 5  # 5D NMR data
    FORMAT_6D = 6  # 6D NMR data
    FORMAT_7D = 7  # 7D NMR data
    FORMAT_8D = 8  # 8D NMR data
    FORMAT_SMALL2D = 12  # Small 2D data
    FORMAT_SMALL3D = 13  # Small 3D data
    FORMAT_SMALL4D = 14  # Small 4D data

    # Unit types (SI units)
    SIUNIT_NONE = 0  # No unit
    SIUNIT_ABUNDANCE = 1  # Abundance
    SIUNIT_HZ = 13  # Hertz
    SIUNIT_PPM = 26  # Parts per million
    SIUNIT_SECONDS = 28  # Seconds
    SIUNIT_DECIBEL = 35  # Decibel

    # Parameter value types
    PARMVAL_NONE = -1  # Not JEOL's definition
    PARMVAL_STR = 0  # String value
    PARMVAL_INT = 1  # Integer value
    PARMVAL_FLT = 2  # Float value
    PARMVAL_Z = 3  # Complex value
    PARMVAL_INF = 4  # Infinity value

    # Infinity types
    INF_NEG = 1  # Negative infinity
    INF_MINUS1 = 2  # Minus 1
    INF_ZERO = 3  # Zero
    INF_PLUS1 = 4  # Plus 1
    INF_POS = 5  # Positive infinity

    # Parameter record constants
    JVAL_STRLEN = 16  # Length of string values in parameter records
    PARAM_NAMELEN = 28  # Length of parameter names in records
    PARAM_RECORD_SIZE = 64  # Total bytes per parameter record
    PARAM_NUM_UNITS = 5  # Number of unit structs per parameter
    PARAM_UNIT_SIZE = 2  # Bytes per unit struct (prefix+power, base)

    def __init__(self, filename: str, options: Optional[Dict] = None):
        super().__init__(filename, options)
        self._file = None
        self._dim_sizes = [0] * 8

        self._dim_types = [0] * 8
        self._axis_start = [0.0] * 8
        self._axis_stop = [0.0] * 8

        # Unit information for each dimension
        self._unit_types = [self.SIUNIT_NONE] * 8  # Unit type codes
        self._unit_exps = [0] * 8  # Unit exponents

    def read(self) -> NMRData:
        with open(self.filename, "rb") as self._file:
            self._parse_header()
            self._params = self._parse_params()
            data = self._read_data()
            dimensions = self._create_dimensions()
            metadata = self._create_metadata()

            return NMRData(
                data=data,
                dimensions=dimensions,
                metadata=metadata,
                source_format="delta",
                source_filename=self.filename,
            )

    def _parse_header(self) -> None:
        try:
            # Read the standard header buffer
            self._file.seek(0)
            hdr_buff = self._file.read(4096)

            # Verify we read the expected header size
            if len(hdr_buff) < 4096:
                raise IOError(
                    f"Expected to read 4096 bytes for header, but got {
                        len(hdr_buff)} bytes")

            # Read endian flag
            endian_flag = struct.unpack_from(
                "B", hdr_buff, self.OFFSET_ENDIAN_FLAG)[0]
            if endian_flag == 0:
                self._byte_order = "big"
            elif endian_flag == 1:
                self._byte_order = "little"
            else:
                raise ValueError(
                    f"Invalid endian flag: {endian_flag}. Expected 0 (big-endian) or 1 (little-endian)")

            # Read dimension count
            self._dim_count = struct.unpack_from(
                "B", hdr_buff, self.OFFSET_DIM_COUNT)[0]
            if self._dim_count < 1 or self._dim_count > self.MAX_DIMENSIONS:
                raise ValueError(
                    f"Invalid dimension count: {self._dim_count}. Expected 1-{self.MAX_DIMENSIONS}"
                )

            # Read data type (2 bits for type code, 7th bit for precision)
            data_type_byte = struct.unpack_from(
                "B", hdr_buff, self.OFFSET_DATA_TYPE)[0]
            # Check 7th bit for float32 precision
            self._data_type = (
                np.float32 if (
                    data_type_byte & self.PRECISION_BIT) else np.float64)

            # Read dimension types
            for i in range(self._dim_count):
                offset = self.OFFSET_DIM_TYPES + i
                self._dim_types[i] = struct.unpack_from(
                    "B", hdr_buff, offset)[0]

            # Read unit information (JEOL_JUNIT structure)
            # Each jUnit structure contains: unitType (1 byte), unitExp (1
            # byte), scaleType (1 byte)
            OFFSET_DATA_UNITS = 32
            for i in range(self._dim_count):
                offset = OFFSET_DATA_UNITS + i * 3
                self._unit_types[i] = struct.unpack_from(
                    "B", hdr_buff, offset)[0]
                self._unit_exps[i] = struct.unpack_from(
                    "B", hdr_buff, offset + 1)[0]
                # scaleType is unused for now

            # Read dimension sizes (always big-endian)
            for i in range(self._dim_count):
                offset = self.OFFSET_DIM_SIZES + i * 4
                self._dim_sizes[i] = struct.unpack_from(
                    ">I", hdr_buff, offset)[0]

            # Read axis start values (always big-endian)
            for i in range(self._dim_count):
                offset = self.OFFSET_AXIS_START + i * 8
                self._axis_start[i] = struct.unpack_from(
                    ">d", hdr_buff, offset)[0]

            # Read axis stop values (always big-endian)
            for i in range(self._dim_count):
                offset = self.OFFSET_AXIS_STOP + i * 8
                self._axis_stop[i] = struct.unpack_from(
                    ">d", hdr_buff, offset)[0]

            # Read base frequency (always big-endian)
            self._base_freq = [0.0] * self.MAX_DIMENSIONS
            fmt_float64 = ">d"  # Always use big-endian
            for i in range(self._dim_count):
                offset = self.OFFSET_BASE_FREQ + i * 8
                self._base_freq[i] = struct.unpack_from(
                    fmt_float64, hdr_buff, offset)[0]

            # Read zero point (offset) (always big-endian)
            self._zero_point = [0.0] * self.MAX_DIMENSIONS
            for i in range(self._dim_count):
                offset = self.OFFSET_ZERO_POINT + i * 8
                self._zero_point[i] = struct.unpack_from(
                    fmt_float64, hdr_buff, offset)[0]

            # Read parameter start position (always big-endian)
            self._param_start = struct.unpack_from(
                ">I", hdr_buff, self.OFFSET_PARAM_START
            )[0]

            # Read parameter length (always big-endian)
            self._param_length = struct.unpack_from(
                ">I", hdr_buff, self.OFFSET_PARAM_LENGTH
            )[0]

            # Read data start position (always big-endian)
            self._data_start = struct.unpack_from(
                ">I", hdr_buff, self.OFFSET_DATA_START
            )[0]

            # Read data length (always big-endian)
            self._data_length = struct.unpack_from(
                ">Q", hdr_buff, self.OFFSET_DATA_LENGTH
            )[0]

        except Exception as e:
            raise IOError(f"Failed to parse header: {str(e)}") from e

    def _parse_params(self) -> Dict:
        """Parse parameters from the JEOL Delta format file.

        Each parameter record is 64 bytes with the following layout:
            Bytes 0-3:   skip (4 bytes)
            Bytes 4-5:   scaler (int16, file endianness)
            Bytes 6-15:  5 unit structs (2 bytes each: prefix+power, base)
            Bytes 16-31: value data (16 bytes, format depends on type)
            Bytes 32-35: value type code (int32, file endianness)
            Bytes 36-63: parameter name (28 bytes, space-padded ASCII)

        Returns:
            Dict: A flat dictionary mapping lowercase parameter names to values.
        """
        params = {}

        try:
            if not hasattr(self, "_param_start") or not hasattr(
                self, "_param_length"
            ):
                self._parse_header()

            if self._param_length <= 0:
                return params

            self._file.seek(self._param_start)
            param_data = self._file.read(self._param_length)

            if len(param_data) != self._param_length:
                raise IOError(
                    f"Expected {self._param_length} bytes for parameters, "
                    f"got {len(param_data)}"
                )

            # Endianness for parameter values follows the file's endian flag
            fmt = "<" if self._byte_order == "little" else ">"

            # Parse parameter section header (4 × uint32)
            parm_size = struct.unpack_from(f"{fmt}I", param_data, 0)[0]
            lo_id = struct.unpack_from(f"{fmt}I", param_data, 4)[0]
            hi_id = struct.unpack_from(f"{fmt}I", param_data, 8)[0]

            num_params = hi_id + 1
            offset = 16  # skip 16-byte header

            for _ in range(num_params):
                if offset + self.PARAM_RECORD_SIZE > len(param_data):
                    break

                rec = param_data[offset: offset + self.PARAM_RECORD_SIZE]

                # Value type at bytes 32-35
                val_type = struct.unpack_from(f"{fmt}i", rec, 32)[0]

                # Value data at bytes 16-31
                val_start = 16
                actual_val = None

                if val_type == self.PARMVAL_STR:
                    raw = rec[val_start: val_start + self.JVAL_STRLEN]
                    # Strip spaces and nulls (matching jeolconverter behavior)
                    actual_val = raw.decode("ascii", errors="replace")
                    actual_val = "".join(
                        c for c in actual_val if c != "\x00" and c != " "
                    )
                elif val_type == self.PARMVAL_INT:
                    actual_val = struct.unpack_from(
                        f"{fmt}i", rec, val_start
                    )[0]
                elif val_type == self.PARMVAL_FLT:
                    actual_val = struct.unpack_from(
                        f"{fmt}d", rec, val_start
                    )[0]
                elif val_type == self.PARMVAL_Z:
                    re_val = struct.unpack_from(
                        f"{fmt}d", rec, val_start
                    )[0]
                    im_val = struct.unpack_from(
                        f"{fmt}d", rec, val_start + 8
                    )[0]
                    actual_val = complex(re_val, im_val)
                elif val_type == self.PARMVAL_INF:
                    inf_code = struct.unpack_from(
                        f"{fmt}i", rec, val_start
                    )[0]
                    inf_map = {
                        self.INF_NEG: float("-inf"),
                        self.INF_POS: float("inf"),
                        self.INF_ZERO: 0.0,
                        self.INF_PLUS1: 1.0,
                        self.INF_MINUS1: -1.0,
                    }
                    actual_val = inf_map.get(inf_code)

                # Parameter name at bytes 36-63 (28 bytes, strip spaces/nulls)
                name_raw = rec[36: 36 + self.PARAM_NAMELEN]
                param_name = (
                    name_raw.decode("ascii", errors="replace")
                    .strip()
                    .strip("\x00")
                )

                if param_name and actual_val is not None:
                    params[param_name.lower()] = actual_val

                offset += self.PARAM_RECORD_SIZE

        except Exception as e:
            import logging
            logging.warning(f"Failed to parse parameters: {str(e)}")

        return params

    def _read_raw_data(self) -> np.ndarray:
        """Read raw data from file and convert to numpy array"""
        try:
            self._file.seek(self._data_start)
            raw_data = self._file.read(self._data_length)

            # Verify that we read the expected amount of data
            if len(raw_data) != self._data_length:
                raise IOError(
                    f"Expected to read {
                        self._data_length} bytes, but got {
                        len(raw_data)} bytes")

            return np.frombuffer(raw_data, dtype=self._data_type)
        except Exception as e:
            raise IOError(
                f"Failed to read raw data from file: {
                    str(e)}") from e

    def _read_data(self) -> np.ndarray:
        data = self._read_raw_data()

        if self._dim_count == 1:
            if self._dim_types[0] == self.DIM_TYPE_REAL:
                return data
            elif self._dim_types[0] == self.DIM_TYPE_COMPLEX:
                # For complex 1D data: [real0, real1, ..., realN, imag0, imag1, ..., imagN]
                # Reshape into complex128 format: [complex0, complex1, ...,
                # complexN]
                return complexify(
                    data, mode="separated", first_component="real")
        elif self._dim_count == 2:
            target_shape = self._dim_sizes[0], self._dim_sizes[1]

            if (
                self._dim_types[0] == self.DIM_TYPE_COMPLEX
                and self._dim_types[1] == self.DIM_TYPE_REAL
            ) or (
                self._dim_types[0] == self.DIM_TYPE_REAL_COMPLEX
                and self._dim_types[1] == self.DIM_TYPE_REAL_COMPLEX
            ):
                # real first then imag, in block
                complex_data = complexify(
                    data, mode="separated", first_component="real"
                )
                return self._delta_reshape(complex_data, target_shape)

            elif (
                self._dim_types[0] == self.DIM_TYPE_COMPLEX
                and self._dim_types[1] == self.DIM_TYPE_COMPLEX
            ):
                # real first then imag, in block
                hypercomplex_size = data.size // 4
                real_real_part = self._delta_reshape(
                    data[:hypercomplex_size], target_shape
                )
                real_imag_part = self._delta_reshape(
                    data[hypercomplex_size: hypercomplex_size * 2], target_shape
                )
                imag_real_part = self._delta_reshape(
                    data[hypercomplex_size * 2: hypercomplex_size * 3], target_shape
                )
                imag_imag_part = self._delta_reshape(
                    data[hypercomplex_size * 3:], target_shape
                )
                dim2_real = real_real_part + 1j * real_imag_part
                dim2_imag = imag_real_part + 1j * imag_imag_part
                return np.concatenate((dim2_real, dim2_imag), axis=1)

        # Fallback for unsupported dimension counts or types
        return data

    def _create_dimensions(self) -> List[DimensionInfo]:
        dimensions = []

        # Create dimension objects for each dimension
        for i in range(self._dim_count):
            size = self._dim_sizes[i]
            start = self._axis_start[i]
            stop = self._axis_stop[i]
            dim_type = self._dim_types[i]

            # Determine if this is complex data
            is_complex = dim_type in [
                self.DIM_TYPE_COMPLEX,
                self.DIM_TYPE_TPPI,
                self.DIM_TYPE_REAL_COMPLEX,
            ]

            is_time_domain = False
            if self._unit_types[i] == self.SIUNIT_SECONDS:
                # Time domain if unit type is seconds with exponent 0 or 1
                if self._unit_exps[i] == 0 or self._unit_exps[i] == 1:
                    is_time_domain = True
            elif self._unit_types[i] == self.SIUNIT_HZ:
                # Hz can be both time and frequency domain, use additional
                # checks
                if i == 0:  # First dimension is more likely to be time domain
                    is_time_domain = True
                if start == 0.0:  # Time typically starts at 0
                    is_time_domain = True
            elif self._unit_types[i] == self.SIUNIT_PPM:
                # PPM is always frequency domain
                is_time_domain = False
            else:
                # Fallback for unknown unit types
                if i == 0:  # First dimension is usually time domain (F2)
                    is_time_domain = True
                if start == 0.0:  # Time typically starts at 0
                    is_time_domain = True
                if (
                    self._base_freq[i] != 0 and self._zero_point[i] != 0
                ):  # Has valid NMR parameters
                    is_time_domain = True

            # Determine unit based on domain type
            # Time domain data axis uses seconds, frequency domain uses ppm
            # Note: transmitter_offset is in Hz regardless of domain type
            unit = "s" if is_time_domain else "ppm"

            # Determine dimension name (F2, F1, F3, etc.) - F2 is first
            # dimension
            dim_name = f"F{self._dim_count - i}"

            # Calculate spectral width correctly based on domain type
            # For time domain: sw = n/t (where n is number of points, t is total time)
            # For frequency domain: sw = abs(stop - start)
            if is_time_domain:
                # Time domain: spectral width is inverse of time increment
                # This is the standard way to calculate spectral width in NMR
                total_time = stop - start
                spectral_width = size / total_time if total_time != 0 else 0.0
            else:
                # Frequency domain: use the axis difference directly
                spectral_width = abs(stop - start)

            # Calculate offset correctly based on domain type
            if is_time_domain:
                # For time domain data, use zero_point to calculate offset
                # (carrier frequency in Hz)
                offset = spectral_width * self._zero_point[i]

            else:
                # For frequency domain data, use axis_start directly
                offset = start

            # Calculate step size for linear axis
            step = (stop - start) / size if size > 0 else 0.0

            dim_info = DimensionInfo(
                size=size,
                is_complex=is_complex,
                spectral_width=spectral_width,
                observation_frequency=self._base_freq[
                    i
                ],  # Store base_freq as observation frequency in MHz
                unit=unit,
                transmitter_offset=offset,
                axis_generator=LinearGenerator(start=start, step=step),
                domain_type="time" if is_time_domain else "frequency",
                can_ft=is_time_domain,  # Can perform FT on time domain data
                domain_metadata={"name": dim_name, "dimension_type": dim_type},
            )

            dimensions.append(dim_info)

        return dimensions

    def _calculate_digital_filter_group_delay(self) -> float:
        """Calculate group delay from JEOL multi-stage decimation filter params.

        Uses the 'orders' and 'factors' parameters to compute:
            sum((order_i - 1) / product(factors[i:])) / 2
        then converts to data points via x_sweep, x_acq_time, x_points.

        Returns:
            Group delay in data points, or 0.0 if parameters are missing.
        """
        p = self._params
        orders_str = p.get("orders")
        factors_str = p.get("factors")
        x_sweep = p.get("x_sweep")
        x_acq_time = p.get("x_acq_time")
        x_points = p.get("x_points")

        if not all(
            v is not None
            for v in [orders_str, factors_str, x_sweep, x_acq_time, x_points]
        ):
            return 0.0

        if not isinstance(orders_str, str) or not isinstance(factors_str, str):
            return 0.0

        try:
            num_stages = int(orders_str[0])
            if num_stages <= 0 or len(orders_str) < 2:
                return 0.0

            orders_rest = orders_str[1:]
            chars_per_order = len(orders_rest) // num_stages
            if chars_per_order <= 0:
                return 0.0

            factors = [int(factors_str[i]) for i in range(num_stages)]
            orders = []
            pos = 0
            for _ in range(num_stages):
                orders.append(int(orders_rest[pos: pos + chars_per_order]))
                pos += chars_per_order

            # Accumulate group delay in abstract units
            total = 0.0
            for stage in range(num_stages):
                product = 1
                for j in range(stage, num_stages):
                    product *= factors[j]
                total += (orders[stage] - 1) / product

            total /= 2.0

            # Convert to data points.
            # When parameters are consistent (x_sweep * x_acq_time ≈ x_points),
            # the conversion refines the value. For arrayed/2D experiments where
            # the parameter set may refer to the indirect dimension, the raw
            # accumulator is already in output data-point units, so use it
            # directly as fallback.
            if x_sweep != 0 and x_acq_time != 0 and x_points != 0:
                expected_points = x_sweep * x_acq_time
                if 0.5 < expected_points / x_points < 2.0:
                    return total / x_sweep / x_acq_time * (x_points - 1)

            return total

        except (ValueError, IndexError):
            return 0.0

    def _create_metadata(self) -> Dict[str, any]:
        group_delay = self._calculate_digital_filter_group_delay()

        metadata = {
            # Basic file information
            "source_format": "delta",
            "source_filename": self.filename,
            "endianness": self._byte_order,
            # Data structure information
            "dimension_count": self._dim_count,
            "data_type": self._data_type.__name__,
            "data_start_offset": self._data_start,
            "data_length_bytes": self._data_length,
            # Dimension-specific information
            "dimension_types": self._dim_types[: self._dim_count],
            "dimension_sizes": self._dim_sizes[: self._dim_count],
            "axis_start_values": self._axis_start[: self._dim_count],
            "axis_stop_values": self._axis_stop[: self._dim_count],
            # NMR-specific parameters
            "base_frequencies": self._base_freq[: self._dim_count],
            "zero_points": self._zero_point[: self._dim_count],
            # Digital filter
            "digital_filter_group_delay": group_delay,
            # Parsed parameters
            "parameters": self._params,
        }

        return metadata

    def _delta_reshape(
            self,
            arr_1d: np.ndarray,
            target_shape: tuple) -> np.ndarray:
        dim_0_size, dim_1_size = target_shape

        # Block-based rearrangement only when both dimensions are
        # divisible by BLOCK_SIZE (standard 2D format). Small_2D and
        # arrayed experiments store data linearly.
        if (dim_0_size % self.BLOCK_SIZE == 0
                and dim_1_size % self.BLOCK_SIZE == 0
                and dim_0_size >= self.BLOCK_SIZE
                and dim_1_size >= self.BLOCK_SIZE):
            return (
                arr_1d.reshape(
                    dim_1_size // self.BLOCK_SIZE,
                    dim_0_size // self.BLOCK_SIZE,
                    self.BLOCK_SIZE,
                    self.BLOCK_SIZE,
                )
                .transpose(1, 3, 0, 2)
                .reshape(dim_0_size, dim_1_size)
            )

        # Linear reshape for small/arrayed data
        return arr_1d.reshape(dim_0_size, dim_1_size)
