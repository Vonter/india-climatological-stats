import os
import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from bs4 import BeautifulSoup, Tag
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import re

@dataclass
class StationMetadata:
    latitude: float
    longitude: float
    elevation: float
    district: str
    state: str
    controlling_rmc: str
    controlling_mc: str

    @classmethod
    def from_soup(cls, soup: BeautifulSoup) -> 'StationMetadata':
        """Extract station metadata from BeautifulSoup object."""
        # Helper function to extract field value
        def get_field(field_name: str) -> str:
            field_element = soup.find(string=lambda text: field_name in str(text))
            if not field_element:
                logging.debug(f"Field '{field_name}' not found in metadata")
                return ""
            # Get the full text of the parent div and split by ':'
            full_text = field_element.parent.get_text().strip()
            parts = full_text.split(':')
            if len(parts) < 2:
                logging.debug(f"Field '{field_name}' found but format unexpected: '{full_text}'")
                return ""
            value = parts[1].strip()
            logging.debug(f"Extracted {field_name}: '{value}'")
            return value

        def clean_numeric(value: str) -> float:
            """Convert string values to float, handling degree formats and cleaning input."""
            if not value or set(value) <= {'-'}:
                return float('nan')
            
            # Handle degree/minute format for lat/long
            if 'deg' in value.lower() or '°' in value:
                try:
                    # Replace various possible degree and minute symbols
                    value = value.replace('deg', '°').replace('&deg;', '°')
                    value = value.replace("'", "'").replace('′', "'").replace('&rsquo;', "'")
                    
                    # Extract direction (N/S/E/W)
                    direction = ''
                    for dir_char in ['N', 'S', 'E', 'W']:
                        if dir_char in value.upper():
                            direction = dir_char
                            value = value.upper().replace(dir_char, '').strip()
                            break
                    
                    # Split by degree symbol
                    parts = value.split('°')
                    degrees = float(parts[0].strip())
                    
                    # Handle minutes if present
                    minutes = 0
                    if len(parts) > 1:
                        minute_part = ''.join(c for c in parts[1] if c.isdigit() or c == '.')
                        if minute_part:
                            minutes = float(minute_part)
                    
                    # Calculate decimal degrees
                    decimal_degrees = degrees + (minutes / 60)
                    
                    # Apply direction
                    if direction in ['S', 'W']:
                        decimal_degrees = -decimal_degrees
                    
                    return decimal_degrees
                    
                except (ValueError, IndexError) as e:
                    logging.warning(f"Failed to parse coordinate value: {value}. Error: {e}")
                    return float('nan')
            
            # For elevation and other numeric values
            try:
                numeric_str = ''.join(c for c in value if c.isdigit() or c in '.-')
                return float(numeric_str)
            except ValueError:
                return float('nan')

        # Extract fields
        fields = {
            "lat": get_field("Latitude"),
            "lon": get_field("Longitude"),
            "elev": get_field("Elevation"),
        }
        
        # Parse numeric values
        parsed_fields = {
            "latitude": clean_numeric(fields["lat"]),
            "longitude": clean_numeric(fields["lon"]),
            "elevation": clean_numeric(fields["elev"]),
        }
        
        # Create and return the object
        return cls(
            **parsed_fields,
            district=get_field("District"),
            state=get_field("State/UT"),
            controlling_rmc=get_field("Controlling RMC"),
            controlling_mc=get_field("Controlling MC")
        )

@dataclass
class ExtremeWeather:
    month: str
    rank: int
    value: float
    date: str
    type: str  # Added type field to distinguish between different extreme types

    @classmethod
    def from_table_cell(cls, month: str, rank: int, cell_text: str, type_name: str) -> Optional['ExtremeWeather']:
        if not cell_text or cell_text.strip() in ['-', '']:
            return None
        
        try:
            # Check if the cell contains a value and date in parentheses
            if '(' in cell_text and ')' in cell_text:
                value_part = cell_text.split('(')[0].strip()
                date_part = cell_text.split('(')[1].rstrip(')')
                
                # Clean the value part to ensure it's a valid float
                # Improved value parsing to handle different formats
                value_part = re.sub(r'[^\d.-]', '', value_part)
                
                # Handle empty value
                if not value_part:
                    value = float('nan')
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        logging.warning(f"Could not convert '{value_part}' to float")
                        return None
                
                # Convert date format from DD-MM-YYYY to YYYY-MM-DD
                try:
                    if re.match(r'\d{1,2}-\d{1,2}-\d{4}', date_part):
                        day, month, year = date_part.split('-')
                        # Ensure day and month are two digits
                        day = day.zfill(2)
                        month = month.zfill(2)
                        date_part = f"{year}-{month}-{day}"
                except Exception as e:
                    logging.warning(f"Failed to convert date format for '{date_part}': {e}")
                    
                return cls(
                    month=month,
                    rank=rank,
                    value=value,
                    date=date_part,
                    type=type_name
                )
            else:
                logging.warning(f"Cell text does not contain expected format: '{cell_text}'")
                return None
            
        except (IndexError, ValueError) as e:
            logging.warning(f"Failed to parse extremes cell: '{cell_text}'. Error: {e}")
            return None

@dataclass
class WeatherStation:
    number: str
    name: str
    metadata: Optional[StationMetadata] = None
    extremes: List[ExtremeWeather] = None

    def __post_init__(self):
        if self.extremes is None:
            self.extremes = []

    @classmethod
    def from_soup(cls, soup: BeautifulSoup) -> 'WeatherStation':
        stn_select = soup.find('select', {'class': 'form-control', 'name': 'stn'})
        if not stn_select or not (selected_option := stn_select.find('option', selected=True)):
            raise ValueError("Could not find station information in HTML")
            
        return cls(
            number=selected_option['value'],
            name=cls._clean_station_name(selected_option.text.strip())
        )
    
    @staticmethod
    def _clean_station_name(name: str) -> str:
        return name.split('[')[0].strip()

@dataclass
class WeatherRecord:
    """Container for weather data associated with a station and month."""
    month: str
    station: WeatherStation
    measurements: Dict[str, Union[float, str]]
    month_extremes: List[ExtremeWeather] = None
    
    def __post_init__(self):
        if self.month_extremes is None:
            self.month_extremes = []

class WeatherDataParser:
    """
    Parser for weather data HTML files.
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize the parser.
        
        Args:
            base_dir: Base directory containing the raw data and where output will be saved
        """
        self.base_dir = base_dir
        self.setup_logging()
        self.stations = {}
        self.max_workers = min(32, os.cpu_count() + 4)  # Optimize thread count
    
    def setup_logging(self) -> None:
        log_dir = self.base_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / 'parse.log'
        if log_file.exists():
            log_file.unlink()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def parse_all_files(self) -> List[WeatherRecord]:
        self._parse_metadata_files()
        self._parse_extremes_files()
        return self._parse_weather_files()
    
    def _parse_metadata_files(self) -> None:
        metadata_files = list((self.base_dir / 'raw').glob('**/metadata.html'))
        logging.info(f"Found {len(metadata_files)} metadata files to process")
        
        # Process metadata files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(executor.map(self._process_metadata_file, enumerate(metadata_files, 1)))
    
    def _process_metadata_file(self, args):
        i, file_path = args
        try:
            station_number = file_path.parent.name
            
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                metadata = StationMetadata.from_soup(soup)
                
                # Get station name from the soup if possible
                station_name = ""
                stn_select = soup.find('select', {'class': 'form-control', 'name': 'stn'})
                if stn_select and (selected_option := stn_select.find('option', selected=True)):
                    station_name = WeatherStation._clean_station_name(selected_option.text.strip())
            
            logging.info(f"Processing metadata file for {station_name} ({station_number})")
            
            # Thread-safe update of stations dictionary
            if station_number not in self.stations:
                self.stations[station_number] = WeatherStation(number=station_number, name=station_name)
            else:
                # Update the name if it was empty before
                if not self.stations[station_number].name and station_name:
                    self.stations[station_number].name = station_name
            self.stations[station_number].metadata = metadata
            
        except Exception as e:
            logging.error(f"Error processing metadata file {file_path}: {e}", exc_info=True)

    def _parse_extremes_files(self) -> None:
        extreme_files = list((self.base_dir / 'raw').glob('**/extreme*.html'))
        logging.info(f"Found {len(extreme_files)} extremes files to process")
        
        # Process extremes files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(executor.map(self._process_extreme_file, enumerate(extreme_files, 1)))
    
    def _process_extreme_file(self, args):
        i, file_path = args
        try:
            station_number = file_path.parent.name
            
            # Extract extreme type from filename
            filename = file_path.name
            extreme_type = self._determine_extreme_type(filename)
            
            if not extreme_type:
                logging.warning(f"Could not determine extreme type from filename: {filename}")
                return
            
            logging.debug(f"Processing extremes file {filename} with type: {extreme_type}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                
                # Get station name from the soup
                station_name = ""
                stn_select = soup.find('select', {'class': 'form-control', 'name': 'stn'})
                if stn_select and (selected_option := stn_select.find('option', selected=True)):
                    station_name = WeatherStation._clean_station_name(selected_option.text.strip())
                
                logging.info(f"Processing extremes file for {station_name} ({station_number}/{filename})")
                
                extreme_data = self._parse_extreme_table(soup, extreme_type)
            
            # Thread-safe update of stations dictionary
            if station_number not in self.stations:
                logging.debug(f"Creating new station entry for {station_number}")
                self.stations[station_number] = WeatherStation(number=station_number, name=station_name)
            else:
                # Update the name if it was empty before
                if not self.stations[station_number].name and station_name:
                    self.stations[station_number].name = station_name
            
            # Append extremes data
            self.stations[station_number].extremes.extend(extreme_data)
                
        except Exception as e:
            logging.error(f"Error processing extremes file {file_path}: {e}", exc_info=True)

    def _determine_extreme_type(self, filename: str) -> str:
        """Determine the extreme weather type from the filename."""
        # Map of possible filename patterns to extreme types
        type_mapping = {
            'extreme_rf': 'rf',               # Rainfall
            'extreme_rain': 'rf',             # Alternative rainfall naming
            'extreme_tmax_high': 'tmax_high', # Maximum temperature (high)
            'extreme_tmax_low': 'tmax_low',   # Maximum temperature (low)
            'extreme_tmin_high': 'tmin_high', # Minimum temperature (high)
            'extreme_tmin_low': 'tmin_low',   # Minimum temperature (low)
        }
        
        # Try exact matches first
        for pattern, extreme_type in type_mapping.items():
            if pattern in filename.lower():
                return extreme_type
        
        # If no exact match, try to infer from the filename
        filename_lower = filename.lower()
        if 'rf' in filename_lower or 'rain' in filename_lower:
            return 'rf'
        elif 'tmax' in filename_lower:
            if 'high' in filename_lower:
                return 'tmax_high'
            elif 'low' in filename_lower:
                return 'tmax_low'
            else:
                # Default to high if not specified
                return 'tmax_high'
        elif 'tmin' in filename_lower:
            if 'high' in filename_lower:
                return 'tmin_high'
            elif 'low' in filename_lower:
                return 'tmin_low'
            else:
                # Default to low if not specified
                return 'tmin_low'
        
        return None

    def _parse_extreme_table(self, soup: BeautifulSoup, extreme_type: str) -> List[ExtremeWeather]:
        """Parse extreme weather tables from HTML using column indices for months."""
        extreme_data = []
        table_div = soup.find('div', {'class': 'table-responsive'})
        if not table_div:
            logging.warning(f"No table-responsive div found in extreme weather HTML for {extreme_type}")
            return extreme_data

        table = table_div.find('table')
        if not table:
            logging.warning(f"No table found in table-responsive div for {extreme_type}")
            return extreme_data

        # Get all rows
        rows = table.find_all('tr')
        if len(rows) < 3:  # Need at least header rows + one data row
            logging.warning(f"Not enough rows in {extreme_type} table, found {len(rows)}")
            return extreme_data
        
        # Skip header rows and process data rows
        data_rows = rows[2:]  # Skip header rows
        logging.debug(f"Found {len(data_rows)} data rows in {extreme_type} table")
        
        for row_idx, row in enumerate(data_rows, 1):
            cells = row.find_all('td')
            if not cells or len(cells) < 2:
                continue

            try:
                rank = int(cells[0].text.strip())
            except ValueError:
                logging.warning(f"Invalid rank in row {row_idx}: '{cells[0].text.strip()}'")
                continue

            # Process each month column (1-12)
            for month_idx, cell in enumerate(cells[1:13], 1):  # 1-based month index (Jan=1, Feb=2, etc.)
                if month_idx > 12:  # Ensure we don't go beyond 12 months
                    break
                
                cell_text = cell.text.strip()
                
                # Create extreme weather record with month index as a string (e.g., "01" for January)
                month_str = f"{month_idx:02d}"  # Format as 2-digit string: 01, 02, ..., 12
                
                extreme = ExtremeWeather.from_table_cell(month_str, rank, cell_text, extreme_type)
                if extreme:
                    extreme_data.append(extreme)
                    logging.debug(f"Parsed extreme weather for month {month_str}: {extreme}")

        return extreme_data

    def _parse_weather_files(self) -> List[WeatherRecord]:
        html_files = [f for f in self._get_html_files() 
                     if not f.name.startswith('extreme') and f.name != 'metadata.html']
        total_files = len(html_files)
        logging.info(f"Found {total_files} normals files to process")
        records = []
        
        # Process weather files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # Submit all tasks
            for i, file_path in enumerate(html_files, 1):
                future = executor.submit(self._parse_file, file_path, total_files, i)
                futures.append(future)
            
            # Collect results as they complete
            for future in futures:
                result = future.result()
                if result is not None:
                    records.append(result)
        
        return records
    
    def _get_html_files(self) -> List[Path]:
        raw_dir = self.base_dir / 'raw'
        if not raw_dir.exists():
            raise FileNotFoundError(f"Directory '{raw_dir}' not found")
        
        html_files = []
        for subdir in raw_dir.iterdir():
            if subdir.is_dir():
                found_files = list(subdir.glob('*.html'))
                html_files.extend(found_files)
        
        return html_files
    
    def _parse_file(self, file_path: Path, total_files: int, file_num: int) -> Optional[WeatherRecord]:
        try:
            station_number = file_path.parent.name
            month = file_path.stem
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            station = WeatherStation.from_soup(soup)
            
            logging.info(f"Processing normals file for {station.name} ({station_number}/{month}.html)")
            
            # Update station with metadata and extremes if available
            if station.number in self.stations:
                stored_station = self.stations[station.number]
                station.metadata = stored_station.metadata
                station.extremes = stored_station.extremes
                # Keep the name from the stored station if it exists
                if stored_station.name and not station.name:
                    station.name = stored_station.name
            
            measurements = self._parse_measurements(soup)

            # Get all extremes for this station and filter by month
            month_extremes = []
            if station.extremes:
                # Pad single digit months with leading zero for consistent comparison
                month_padded = month.zfill(2)
                # Filter extremes for this month and add to list
                month_extremes = [
                    extreme for extreme in station.extremes 
                    if extreme.month == month_padded
                ]
            else:
                # If no extremes data available, initialize empty list
                month_extremes = []
            
            return WeatherRecord(
                month=month,
                station=station,
                measurements=measurements,
                month_extremes=month_extremes
            )
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}", exc_info=True)
            return None
    
    def _parse_measurements(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract weather measurements from the HTML soup."""
        measurements = {}
        card_body = soup.find('div', {'class': 'card-body'})
        
        if not card_body:
            logging.warning("No card-body found in HTML")
            return measurements
            
        # Process all tables at once
        tables = card_body.find_all('table')
        for table in tables:
            self._parse_table(table, measurements)
            
        return measurements
    
    def _parse_table(self, table: Tag, measurements: Dict[str, Any]) -> None:
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all(['th', 'td'])
            if len(cols) < 2:
                continue
                
            key = self._clean_key(cols[0].text.strip())
            if not key:
                continue
                
            if len(cols) == 3:  # 3-column format with 03UTC and 12UTC
                value_03utc = self._parse_value(cols[1].text.strip())
                value_12utc = self._parse_value(cols[2].text.strip())
                measurements[f"{key}_03UTC"] = value_03utc
                measurements[f"{key}_12UTC"] = value_12utc
            else:  # 2-column format
                value = self._parse_value(cols[1].text.strip())
                measurements[key] = value
    
    @staticmethod
    def _clean_key(key: str) -> Optional[str]:
        """Clean and normalize measurement keys."""
        key = key.strip()
        
        # Skip header rows
        if key in ["Parameter", "03 UTC"]:
            return None
        
        # Remove unnecessary text
        key = key.replace('Parameter', '').replace('03 UTC', '').strip()
        
        # Replace non-printing characters with underscore
        key = re.sub(r'\s+', '_', key)  # Replace any whitespace with underscore
        
        # Define replacements for better readability
        replacements = [
            ('[', ''), (']', ''),           # Remove brackets
            ('/', '_'), ('-', '_'),         # Replace slashes and hyphens with underscores
            ('.', ''), (',', ''),           # Remove periods and commas
            ('°', ''),                      # Remove degree symbol
            ('%', 'percent'),               # Replace % with 'percent'
        ]
        
        # Apply all replacements
        for old, new in replacements:
            key = key.replace(old, new)
        
        # Convert to lowercase
        key = key.lower()
        
        # Clean up multiple underscores and remove leading/trailing underscores
        while '__' in key:
            key = key.replace('__', '_')
        
        return key.strip('_')
    
    @staticmethod
    def _parse_value(value: str) -> Any:
        value_clean = value.split('(')[0].strip()
        # Return NaN for empty strings or hyphens
        if not value_clean or value_clean == '-' or value_clean == '---':
            return float('nan')
        try:
            return float(value_clean)
        except ValueError:
            return value_clean

class WeatherDataExporter:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
    
    def export_data(self, records: List[WeatherRecord]) -> None:
        self._save_parquet(records)
        self._save_csv(records)
    
    def _create_flat_records(self, records: List[WeatherRecord]) -> List[Dict[str, Any]]:
        """Create flat dictionary records from WeatherRecord objects."""
        flat_data = []
        
        # Define extreme weather types once with their new names
        extreme_types = ['tmax_high', 'tmax_low', 'tmin_high', 'tmin_low', 'rf']
        extreme_type_display_names = {
            'tmax_high': 'max_high_temp',
            'tmax_low': 'max_low_temp',
            'tmin_high': 'min_high_temp',
            'tmin_low': 'min_low_temp',
            'rf': 'max_rainfall'
        }
        
        for record in records:
            # Base record information
            flat_record = {
                'station_number': record.station.number,
                'station_name': record.station.name,
                'month': record.month,
                'month_name': record.month.split('_')[0].capitalize()
            }
            
            # Add metadata fields with renamed elevation field
            if record.station.metadata:
                metadata_dict = asdict(record.station.metadata)
                # Rename elevation to altitude
                if 'elevation' in metadata_dict:
                    metadata_dict['altitude'] = metadata_dict.pop('elevation')
                flat_record.update(metadata_dict)
            else:
                # Add empty metadata fields for consistency with renamed elevation
                flat_record.update({
                    'latitude': float('nan'),
                    'longitude': float('nan'),
                    'altitude': float('nan'),  # Renamed from elevation
                    'district': '',
                    'state': '',
                    'controlling_rmc': '',
                    'controlling_mc': ''
                })
            
            # Add measurements
            flat_record.update(record.measurements)
            
            # Initialize extreme fields with NaN/empty values
            for ext_type in extreme_types:
                display_name = extreme_type_display_names[ext_type]
                
                # For Parquet, we'll store all ranks (1-10)
                for rank in range(1, 11):
                    flat_record[f"{display_name}_rank{rank}_value"] = float('nan')
                    flat_record[f"{display_name}_rank{rank}_date"] = ''
                
                # Also keep the rank 1 fields with the new naming for CSV compatibility
                flat_record[f"{display_name}_value"] = float('nan')
                flat_record[f"{display_name}_date"] = ''
            
            # Fill in extreme values that exist
            for extreme in record.month_extremes or []:
                if extreme.rank <= 10:  # Include ranks 1-10
                    display_name = extreme_type_display_names[extreme.type]
                    flat_record[f"{display_name}_rank{extreme.rank}_value"] = extreme.value
                    flat_record[f"{display_name}_rank{extreme.rank}_date"] = extreme.date
                    
                    # Also update the rank 1 fields with the new naming for CSV compatibility
                    if extreme.rank == 1:
                        flat_record[f"{display_name}_value"] = extreme.value
                        flat_record[f"{display_name}_date"] = extreme.date
            
            flat_data.append(flat_record)
        
        return flat_data
    
    def _save_parquet(self, records: List[WeatherRecord]) -> None:
        """Save weather data in Parquet format with a flat, easy-to-use structure."""
        output_file = self.base_dir / 'climatological.parquet'
        try:
            logging.info(f"Preparing {len(records)} records for Parquet export")
            flat_data = self._create_flat_records(records)
            
            # Convert to pandas DataFrame and save
            pd.DataFrame(flat_data).to_parquet(
                output_file, 
                compression='snappy',
                index=False,
                row_group_size=100000
            )
            
            logging.info(f"Successfully saved Parquet data to {output_file}")
        except Exception as e:
            logging.error(f"Error saving Parquet file: {e}", exc_info=True)
    
    def _save_csv(self, records: List[WeatherRecord]) -> None:
        """Save weather data in CSV format with specific columns in a defined order."""
        output_file = self.base_dir / 'climatological.csv'
        
        try:
            # Reuse the same flat data structure created for Parquet
            flat_data = self._create_flat_records(records)
            
            # Define the exact columns and order for CSV with renamed columns
            csv_columns = [
                'station_name',
                'month',
                'latitude',
                'longitude',
                'altitude',  # Renamed from elevation
                'mean_daily_maximum_temperature_in_c',
                'mean_daily_minimum_temperature_in_c',
                'max_high_temp_value',  # Renamed from tmax_high_value
                'max_high_temp_date',   # Renamed from tmax_high_date
                'max_low_temp_value',   # Renamed from tmax_low_value
                'max_low_temp_date',    # Renamed from tmax_low_date
                'min_high_temp_value',  # Renamed from tmin_high_value
                'min_high_temp_date',   # Renamed from tmin_high_date
                'min_low_temp_value',   # Renamed from tmin_low_value
                'min_low_temp_date',    # Renamed from tmin_low_date
                'mean_monthly_total_rainfall_in_mm',
                'mean_monthly_rainy_days',
                'max_rainfall_value',   # Renamed from rf_value
                'max_rainfall_date',    # Renamed from rf_date
                'mean_of_highest_maximum_temperature_in_the_month_in_c',
                'mean_of_lowest_minimum_temperature_in_the_month_in_c',
                'mean_dry_bulb_temperature_in_c_03UTC',
                'mean_dry_bulb_temperature_in_c_12UTC',
                'mean_wet_bulb_temperature_in_c_03UTC',
                'mean_wet_bulb_temperature_in_c_12UTC',
                'mean_station_level_pressure_in_hpa_03UTC',
                'mean_station_level_pressure_in_hpa_12UTC',
                'mean_vapour_pressure_in_hpa_03UTC',
                'mean_vapour_pressure_in_hpa_12UTC',
                'mean_relative_humidity_in_percent_03UTC',
                'mean_relative_humidity_in_percent_12UTC',
                'mean_total_clouds_in_okta_03UTC',
                'mean_total_clouds_in_okta_12UTC',
                'mean_low_clouds_in_okta_03UTC',
                'mean_low_clouds_in_okta_12UTC',
                'mean_wind_speed_in_kmph',
                'district',
                'state',
                'controlling_rmc',
                'controlling_mc',
                'station_number'
            ]
            
            # Create a new list with only the specified columns
            csv_data = []
            for record in flat_data:
                csv_record = {}
                for col in csv_columns:
                    csv_record[col] = record.get(col, '')
                csv_data.append(csv_record)
            
            # Sort the data by station_name and month
            csv_data.sort(key=lambda x: (x['station_name'], x['month']))
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerows(csv_data)

            logging.info(f"Successfully saved CSV data to {output_file} with {len(csv_columns)} columns")
        except Exception as e:
            logging.error(f"Error saving CSV file: {e}", exc_info=True)

def main():
    base_dir = Path(__file__).parent
    
    parser = WeatherDataParser(base_dir)
    records = parser.parse_all_files()
    
    exporter = WeatherDataExporter(base_dir)
    exporter.export_data(records)
    
    logging.info("Processing completed")

if __name__ == "__main__":
    main()