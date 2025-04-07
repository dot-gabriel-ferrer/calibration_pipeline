import os
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm

class ObservationManager:
    def __init__(self, base_path):
        """
        Organize calibration files (bias, darks, flats) and field files (VIS/UV).
        We do NOT rename them on disk, just store them in a data structure.
        """
        self.base_path = Path(base_path)
        self.data = {
            'CALIBRATION': {
                'bias': [],
                'darks': [],
                'flats': []
            },
            'FIELDS': {}
        }

    ## ----------------------------------------------------------------
    ##  1) Funciones auxiliares para lectura de FITS y metadatos
    ## ----------------------------------------------------------------

    def _read_fits_header(self, fits_path):
        """Return the primary FITS header or None if unreadable."""
        try:
            with fits.open(fits_path) as hdul:
                return hdul[0].header
        except Exception:
            return None

    def _extract_temperature(self, hdr, file_path=None):
        """
        Extrae la temperatura del header. Ajusta 'TEMP' según la keyword real.
        Si no existe, actualmente devuelve None (y descartará el archivo).
        """
        if hdr is None:
            return None
        temp_val = hdr.get('TEMP')
        if temp_val is None:
            print(f"[WARNING] 'TEMP' keyword not found in {file_path}")
            # Opcional: return 0.0 if no 'TEMP' => para no descartarlo
            return None  # si devuelves None => se descarta
        return float(temp_val)

    def _extract_exposure(self, hdr):
        """Extrae EXPTIME del header FITS."""
        if hdr is None:
            return None
        return float(hdr.get('EXPTIME', 0.0))

    def _determine_vis_or_uv(self, filename, hdr):
        """
        Decide si un archivo de campo es 'vis' o 'uv', p.ej. si
        - el filename incluye 'uv' => uv
        - el filename incluye 'vis' => vis
        - si EXPTIME <= 1.5 => uv
        - else => vis
        """
        fname_lower = filename.lower()
        if 'uv' in fname_lower:
            return 'uv'
        elif 'vis' in fname_lower:
            return 'vis'
        exp_time = self._extract_exposure(hdr)
        if exp_time is not None and exp_time <= 1.5:
            return 'uv'
        return 'vis'

    def _generate_filename(self, category, subcat=None, extension='fits', counter=0,
                           field_name=None, mode=None):
        """
        Genera un nombre interno (identifier) para cada archivo.
        No renombra en disco, solo lo guardamos en self.data.
        """
        if category == 'CALIBRATION':
            # subcat in ['bias','darks','flats']
            return f"{subcat}_{counter:04d}.{extension}"
        elif category == 'FIELDS':
            field_name = field_name or 'unknownfield'
            mode = mode or 'vis'
            return f"field_{field_name}_{mode}_{counter:04d}.{extension}"
        else:
            return f"unknown_{counter:04d}.{extension}"

    def _extract_field_name(self, path):
        return path.name

    ## ----------------------------------------------------------------
    ##  2) load_and_organize(): recorre carpetas y clasifica
    ## ----------------------------------------------------------------
    def load_and_organize(self):
        """
        Recorre self.base_path, detecta calibraciones vs. fields, y
        clasifica cada archivo .fits en self.data.
        """
        # Contadores por subcategoría
        calib_counters = {
            'bias': 1,
            'darks': 1,
            'flats': 1
        }
        field_counters = {}  # p.ej. {'field_1': {'vis':1,'uv':1}}

        for root, dirs, files in tqdm(os.walk(self.base_path), desc="Organizing files:"):
            current_path = Path(root).resolve()
            path_str = str(current_path).lower()

            # 1) Decide si es calibración o field
            #    a) busco 'flat' con prioridad
            #    b) sino 'bias'
            #    c) sino 'dark'
            #    d) sino 'field/campo'
            #    e) sino skip
            subcat_calib = None
            if 'flat' in path_str or 'enfocad' in path_str:
                subcat_calib = 'flats'
            elif 'bias' in path_str:
                subcat_calib = 'bias'
            elif 'dark' in path_str:
                subcat_calib = 'darks'
            elif any(k in path_str for k in ['field','campo']):
                subcat_calib = 'FIELDS'
            else:
                # Ni calibración ni field => skip
                continue

            # 2) Recorre los archivos
            for filename in files:
                if not filename.lower().endswith('.fits'):
                    continue

                file_path = current_path / filename
                hdr = self._read_fits_header(file_path)
                temp = self._extract_temperature(hdr, file_path)
                if temp is None:
                    # Ojo: si quieres no descartarlos, pon un temp=0.0
                    continue
                exp_time = self._extract_exposure(hdr)

                # 3) Calibración
                if subcat_calib in ['bias','darks','flats']:
                    index = calib_counters[subcat_calib]
                    ident = self._generate_filename('CALIBRATION', subcat=subcat_calib,
                                                    extension='fits', counter=index)
                    calib_counters[subcat_calib] += 1

                    self.data['CALIBRATION'][subcat_calib].append({
                        'original_path': str(file_path),
                        'original_name': filename,
                        'identifier': ident,
                        'temperature': temp,
                        'exposure': exp_time
                    })

                # 4) Campos
                elif subcat_calib == 'FIELDS':
                    # Determina el nombre del field
                    field_name = self._extract_field_name(current_path)
                    if field_name not in self.data['FIELDS']:
                        self.data['FIELDS'][field_name] = {'vis': [], 'uv': []}
                        field_counters[field_name] = {'vis': 1, 'uv': 1}

                    mode = self._determine_vis_or_uv(filename, hdr)
                    idx = field_counters[field_name][mode]
                    ident = self._generate_filename('FIELDS',
                                                    field_name=field_name,
                                                    mode=mode,
                                                    extension='fits',
                                                    counter=idx)
                    field_counters[field_name][mode] += 1

                    self.data['FIELDS'][field_name][mode].append({
                        'original_path': str(file_path),
                        'original_name': filename,
                        'identifier': ident,
                        'temperature': temp,
                        'exposure': exp_time
                    })

    ## ----------------------------------------------------------------
    ##  3) filter_files(): filtra por rango de T,Exp, etc.
    ## ----------------------------------------------------------------
    def filter_files(self, 
                     category=None,      # 'CALIBRATION' or 'FIELDS'
                     subcat=None,        # 'bias','darks','flats'
                     field_name=None,    # para fields
                     mode=None,          # 'vis'/'uv' p.ej.
                     temp_min=None,
                     temp_max=None,
                     exp_min=None,
                     exp_max=None,
                     ext_filter=None):
        """Filtra los archivos según categoría, subcat, field, etc."""
        results = []

        def in_range(val, low, high):
            if val is None:
                return False
            if (low is not None and val < low):
                return False
            if (high is not None and val > high):
                return False
            return True

        def match_ext(path):
            if not ext_filter:
                return True
            return Path(path).suffix.lower().lstrip('.') == ext_filter.lower()

        # 1) Calibración
        if category == 'CALIBRATION':
            subcats = [subcat] if subcat else ['bias','darks','flats']
            for sc in subcats:
                for entry in self.data['CALIBRATION'][sc]:
                    Tok = in_range(entry['temperature'], temp_min, temp_max)
                    Eok = in_range(entry['exposure'], exp_min, exp_max)
                    Xok = match_ext(entry['original_path'])
                    if Tok and Eok and Xok:
                        results.append(entry)

        # 2) Fields
        elif category == 'FIELDS':
            fields = [field_name] if field_name else list(self.data['FIELDS'].keys())
            for fld in fields:
                modes = [mode] if mode else ['vis','uv']
                for m in modes:
                    for entry in self.data['FIELDS'][fld][m]:
                        Tok = in_range(entry['temperature'], temp_min, temp_max)
                        Eok = in_range(entry['exposure'], exp_min, exp_max)
                        Xok = match_ext(entry['original_path'])
                        if Tok and Eok and Xok:
                            results.append(entry)
        else:
            # si category=None => buscar en todo
            # 1) calibración
            for sc in ['bias','darks','flats']:
                for entry in self.data['CALIBRATION'][sc]:
                    if (in_range(entry['temperature'], temp_min, temp_max) and
                        in_range(entry['exposure'], exp_min, exp_max) and
                        match_ext(entry['original_path'])):
                        results.append(entry)
            # 2) fields
            for fld in self.data['FIELDS']:
                for m in ['vis','uv']:
                    for entry in self.data['FIELDS'][fld][m]:
                        if (in_range(entry['temperature'], temp_min, temp_max) and
                            in_range(entry['exposure'], exp_min, exp_max) and
                            match_ext(entry['original_path'])):
                            results.append(entry)

        return results

    def get_file_list(self, category=None, subcat=None, field_name=None, mode=None):
        """Devuelve directamente la lista del data structure, sin filtrar."""
        if category == 'CALIBRATION':
            if subcat and subcat in self.data['CALIBRATION']:
                return self.data['CALIBRATION'][subcat]
            else:
                # suma de bias,darks,flats
                combined = []
                for sc in self.data['CALIBRATION']:
                    combined.extend(self.data['CALIBRATION'][sc])
                return combined

        elif category == 'FIELDS':
            if field_name in self.data['FIELDS']:
                if mode in ['vis','uv']:
                    return self.data['FIELDS'][field_name][mode]
                else:
                    return self.data['FIELDS'][field_name]['vis'] + self.data['FIELDS'][field_name]['uv']
            else:
                # all fields
                all_f = []
                for f in self.data['FIELDS']:
                    all_f.extend(self.data['FIELDS'][f]['vis'])
                    all_f.extend(self.data['FIELDS'][f]['uv'])
                return all_f

        return self.data

