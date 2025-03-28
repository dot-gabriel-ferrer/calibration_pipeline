import os
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm

class ObservationManager:
    def __init__(self, base_path):
        """
        Organize calibration files (bias, darks, flats) and
        field files (VIS/UV, grouped by field directories),
        but do NOT rename them on disk. Instead, we store
        a 'generated_id' or 'identifier' in the dictionary 
        for internal reference.
        
        :param base_path: The path to the top-level data directory (e.g. 'data/')
        """
        self.base_path = Path(base_path)

        # Data structure:
        # {
        #   'CALIBRATION': {
        #       'bias': [],
        #       'darks': [],
        #       'flats': []
        #   },
        #   'FIELDS': {
        #       'field1': {
        #           'vis': [],
        #           'uv':  []
        #       },
        #       'field2': {
        #           'vis': [],
        #           'uv':  []
        #       },
        #   }
        # }
        self.data = {
            'CALIBRATION': {
                'bias': [],
                'darks': [],
                'flats': []
            },
            'FIELDS': {}
        }

    def _read_fits_header(self, fits_path):
        """Return the primary FITS header or None if unreadable."""
        try:
            with fits.open(fits_path) as hdul:
                hdr = hdul[0].header
            return hdr
        except Exception:
            return None

    def _extract_temperature(self, hdr, file_path=None):
        if hdr is None:
            return None
        temp_val = hdr.get('TEMP')
        if temp_val is None:
            print(f"[WARNING] 'TEMP' keyword not found in {file_path}")
            return None
        return float(temp_val)

    def _extract_exposure(self, hdr):
        """Extract exposure time from the FITS header (adjust key if needed)."""
        if hdr is None:
            return None
        return float(hdr.get('EXPTIME', None))

    def _determine_vis_or_uv(self, filename, hdr):
        """
        Decide if a field file is 'vis' or 'uv'.
        For example:
         - if the file name contains 'uv', label it 'uv'.
         - if it contains 'vis', label it 'vis'.
         - otherwise, fallback to checking the exposure time or default to 'vis'.
        """
        fname_lower = filename.lower()
        if 'uv' in fname_lower:
            return 'uv'
        elif 'vis' in fname_lower:
            return 'vis'
        else:
            # Optional fallback: check exposure time
            exp_time = self._extract_exposure(hdr)
            if exp_time is not None and exp_time <= 1.5:
                return 'uv'
            return 'vis'

    def _generate_filename(self, category, field_name=None, vis_or_uv=None,
                           subcat=None, extension='fits', counter=0):
        """
        Generate a standardized "identifier" or "name" for internal reference.
        We do NOT rename files on disk; we only store this string in our dictionary.
        
        For calibration: e.g., bias_0001.fits, darks_0002.fits, flats_0003.fits
        For fields: e.g., field_field1_uv_0001.fits
        """
        if category == 'CALIBRATION':
            # subcat can be 'bias', 'darks', 'flats'
            if subcat not in ['bias', 'darks', 'flats']:
                subcat = 'unknown_calib'
            return f"{subcat}_{counter:04d}.{extension}"

        elif category == 'FIELDS':
            if field_name is None:
                field_name = 'unknownfield'
            if vis_or_uv is None:
                vis_or_uv = 'vis'
            return f"field_{field_name}_{vis_or_uv}_{counter:04d}.{extension}"

        else:
            return f"unknown_{counter:04d}.{extension}"

    def _extract_field_name(self, path):
        """
        Determine the name of the field from the directory path.
        For example, if folder is 'data/fields1/', path.name is 'fields1'.
        """
        return path.name  # You can customize how you parse this if needed.

    def load_and_organize(self):
        """
        Walk through directories, detect calibration vs. field, 
        generate an internal identifier for each file, but do NOT rename on disk.
        """
        # Counters for calibration
        calib_counters = {
            'bias': 1,
            'darks': 1,
            'flats': 1,
            'unknown_calib': 1
        }
        # For fields, we keep counters for each (field_name, vis/uv)
        field_counters = {}  # e.g. {'field1': {'vis': X, 'uv': Y}, ...}

        for root, dirs, files in tqdm(os.walk(self.base_path), desc="Organizing files:"):
            current_path = Path(root).resolve()
            path_str = str(current_path).lower()

            # Decide if this path is calibration or field
            if any(k in path_str for k in ['bias', 'dark', 'flat']):
                # It's a calibration folder
                if 'bias' in path_str:
                    subcat = 'bias'
                elif 'dark' in path_str:
                    subcat = 'darks'
                elif 'flat' in path_str:
                    subcat = 'flats'
                else:
                    subcat = 'unknown_calib'

                for filename in tqdm(files, desc=f"Organizing files in {current_path}", leave=False):
                    ext = filename.split('.')[-1].lower()
                    file_path = current_path / filename

                    hdr = None
                    if ext == 'fits':
                        hdr = self._read_fits_header(file_path)

                    temp = self._extract_temperature(hdr, file_path)
                    # Si no hay temperatura, excluimos el archivo
                    if temp is None:
                        continue

                    exp_time = self._extract_exposure(hdr)

                    # El resto de la lógica, por ejemplo:
                    internal_id = self._generate_filename(
                        category='CALIBRATION',
                        subcat=subcat,
                        extension=ext,
                        counter=calib_counters[subcat]
                    )
                    calib_counters[subcat] += 1

                    self.data['CALIBRATION'][subcat].append({
                        'original_path': str(file_path),
                        'original_name': filename,
                        'identifier': internal_id,
                        'temperature': temp,
                        'exposure': exp_time
                    })


            elif any(k in path_str for k in ['field', 'campo']):
                # It's a field folder
                field_name = self._extract_field_name(current_path)
                if field_name not in self.data['FIELDS']:
                    self.data['FIELDS'][field_name] = {'vis': [], 'uv': []}

                if field_name not in field_counters:
                    field_counters[field_name] = {'vis': 1, 'uv': 1}

                for filename in files:
                    ext = filename.split('.')[-1].lower()
                    file_path = current_path / filename

                    hdr = None
                    if ext == 'fits':
                        hdr = self._read_fits_header(file_path)

                    temp = self._extract_temperature(hdr, file_path=file_path)
                    exp_time = self._extract_exposure(hdr)

                    vis_or_uv = self._determine_vis_or_uv(filename, hdr)

                    internal_id = self._generate_filename(
                        category='FIELDS',
                        field_name=field_name,
                        vis_or_uv=vis_or_uv,
                        extension=ext,
                        counter=field_counters[field_name][vis_or_uv]
                    )
                    field_counters[field_name][vis_or_uv] += 1

                    self.data['FIELDS'][field_name][vis_or_uv].append({
                        'original_path': str(file_path),
                        'original_name': filename,
                        'identifier': internal_id,
                        'temperature': temp,
                        'exposure': exp_time
                    })

            else:
                # Not recognized as calibration or field. We do nothing or store them separately.
                pass

    def filter_files(self,
                    category=None,      # 'CALIBRATION' or 'FIELDS'
                    field_name=None,    # e.g. 'field_1'
                    subcat=None,        # e.g. 'bias', 'darks', 'flats'
                    mode=None,          # 'vis' or 'uv'
                    temp_min=None,
                    temp_max=None,
                    exp_min=None,
                    exp_max=None,
                    ext_filter=None):
        """
        Filter files based on:
        - category: 'CALIBRATION' or 'FIELDS'
        - field_name (for category='FIELDS'), e.g. 'field_1'
        - subcat (for category='CALIBRATION'): 'bias', 'darks', 'flats'
        - mode (for category='FIELDS'): 'vis' or 'uv'
        - temp_min, temp_max: numeric temperature range
        - exp_min, exp_max: numeric exposure time range
        - ext_filter: a string for file extension (e.g. 'fits', 'jpg', 'raw');
                        if None, no extension filtering is applied.

        Returns a list of dictionaries representing the files that match.
        """
        results = []

        def in_range(value, low, high):
            """Check if 'value' is within [low, high], ignoring None where appropriate."""
            if value is None:
                return False
            if low is not None and value < low:
                return False
            if high is not None and value > high:
                return False
            return True

        def match_extension(file_path):
            """
            If ext_filter is None, accept all.
            Otherwise, only accept if the file extension matches ext_filter.
            """
            if ext_filter is None:
                return True
            from pathlib import Path
            actual_ext = Path(file_path).suffix.lower().lstrip('.')  # e.g. 'fits'
            return (actual_ext == ext_filter.lower())

        # -------------------
        # CASE 1: CALIBRATION
        # -------------------
        if category == 'CALIBRATION':
            # Gather relevant subcategories
            possible_subcats = list(self.data['CALIBRATION'].keys())  # e.g. ['bias', 'darks', 'flats']
            if subcat in possible_subcats:
                possible_subcats = [subcat]

            for sc in possible_subcats:
                for file_dict in self.data['CALIBRATION'][sc]:
                    temp_ok = ((temp_min is None and temp_max is None) or 
                            in_range(file_dict['temperature'], temp_min, temp_max))
                    exp_ok = ((exp_min is None and exp_max is None) or 
                            in_range(file_dict['exposure'], exp_min, exp_max))
                    ext_ok = match_extension(file_dict['original_path'])

                    if temp_ok and exp_ok and ext_ok:
                        results.append(file_dict)

        # ---------------
        # CASE 2: FIELDS
        # ---------------
        elif category == 'FIELDS':
            # If a specific field name is requested and it exists, use that. Otherwise, use all.
            possible_fields = [field_name] if field_name in self.data['FIELDS'] else list(self.data['FIELDS'].keys())

            for fld in possible_fields:
                # If a specific mode is requested, use it; otherwise, check 'vis' and 'uv'.
                possible_modes = ['vis', 'uv']
                if mode in possible_modes:
                    possible_modes = [mode]

                for m in possible_modes:
                    for file_dict in self.data['FIELDS'][fld][m]:
                        temp_ok = ((temp_min is None and temp_max is None) or 
                                in_range(file_dict['temperature'], temp_min, temp_max))
                        exp_ok = ((exp_min is None and exp_max is None) or 
                                in_range(file_dict['exposure'], exp_min, exp_max))
                        ext_ok = match_extension(file_dict['original_path'])

                        if temp_ok and exp_ok and ext_ok:
                            results.append(file_dict)

        # -------------------
        # CASE 3: ALL DATA
        # -------------------
        else:
            # No category => search all calibrations and all fields.

            # -- Calibrations
            for sc in self.data['CALIBRATION']:
                for file_dict in self.data['CALIBRATION'][sc]:
                    temp_ok = ((temp_min is None and temp_max is None) or 
                            in_range(file_dict['temperature'], temp_min, temp_max))
                    exp_ok = ((exp_min is None and exp_max is None) or 
                            in_range(file_dict['exposure'], exp_min, exp_max))
                    ext_ok = match_extension(file_dict['original_path'])

                    if temp_ok and exp_ok and ext_ok:
                        results.append(file_dict)

            # -- Fields
            for fld in self.data['FIELDS']:
                for m in ['vis', 'uv']:
                    for file_dict in self.data['FIELDS'][fld][m]:
                        temp_ok = ((temp_min is None and temp_max is None) or 
                                in_range(file_dict['temperature'], temp_min, temp_max))
                        exp_ok = ((exp_min is None and exp_max is None) or 
                                in_range(file_dict['exposure'], exp_min, exp_max))
                        ext_ok = match_extension(file_dict['original_path'])

                        if temp_ok and exp_ok and ext_ok:
                            results.append(file_dict)

        return results


    def get_file_list(self, category=None, field_name=None, mode=None, subcat=None):
        """
        Return a list of files from the data structure, optionally filtered by:
         - category: 'CALIBRATION' or 'FIELDS'
         - field_name (only if category='FIELDS')
         - mode: 'vis' or 'uv' (only if category='FIELDS')
         - subcat: 'bias','darks','flats' (only if category='CALIBRATION')
        """
        if category == 'CALIBRATION':
            if subcat and subcat in self.data['CALIBRATION']:
                return self.data['CALIBRATION'][subcat]
            else:
                combined = []
                for sc in self.data['CALIBRATION']:
                    combined.extend(self.data['CALIBRATION'][sc])
                return combined

        elif category == 'FIELDS':
            if field_name and field_name in self.data['FIELDS']:
                if mode in ['vis', 'uv']:
                    return self.data['FIELDS'][field_name][mode]
                else:
                    return self.data['FIELDS'][field_name]['vis'] + self.data['FIELDS'][field_name]['uv']
            else:
                # Return all fields
                all_fields = []
                for fld in self.data['FIELDS']:
                    all_fields.extend(self.data['FIELDS'][fld]['vis'])
                    all_fields.extend(self.data['FIELDS'][fld]['uv'])
                return all_fields

        else:
            # Return everything
            return self.data
