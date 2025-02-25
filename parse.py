import os
import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from bs4 import BeautifulSoup, Tag
import pyarrow as pa
import pyarrow.parquet as pq
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
                return ""
            # Get the full text of the parent div and split by ':'
            full_text = field_element.parent.get_text().strip()
            parts = full_text.split(':')
            if len(parts) < 2:
                return ""
            return parts[1].strip()

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
        
        # Log raw values for debugging
        for key, value in fields.items():
            logging.debug(f"Raw {key}: {value}")
        
        # Parse numeric values
        parsed_fields = {
            "latitude": clean_numeric(fields["lat"]),
            "longitude": clean_numeric(fields["lon"]),
            "elevation": clean_numeric(fields["elev"]),
        }
        
        # Log parsed values
        for key, value in parsed_fields.items():
            logging.debug(f"Parsed {key}: {value}")
        
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
            value_part = cell_text.split('(')[0].strip()
            date_part = cell_text.split('(')[1].rstrip(')')
            value = float(value_part)
            
            return cls(
                month=month,
                rank=rank,
                value=value,
                date=date_part,
                type=type_name
            )
        except (IndexError, ValueError) as e:
            logging.debug(f"Failed to parse extreme weather cell: '{cell_text}'. Error: {e}")
            return None

@dataclass
class WeatherStation:
    number: str
    name: str
    metadata: Optional[StationMetadata] = None
    extreme_weather: List[ExtremeWeather] = None

    def __post_init__(self):
        if self.extreme_weather is None:
            self.extreme_weather = []

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a dictionary for serialization."""
        result = {
            'station_number': self.station.number,
            'station_name': self.station.name,
            'month': self.month
        }
        
        if self.station.metadata:
            result['metadata'] = asdict(self.station.metadata)

        if self.measurements:
            result['measurements'] = self.measurements
            
        if self.station.extreme_weather:
            result['extreme_weather'] = [asdict(ew) for ew in self.station.extreme_weather]
            
        return result

class WeatherDataParser:
    """
    Parser for weather data HTML files.
    
    This class handles parsing of metadata, extreme weather data,
    and regular weather measurement files.
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize the parser.
        
        Args:
            base_dir: Base directory containing the raw data and where output will be saved
        """
        self.base_dir = base_dir
        self.setup_logging()
        self.stations: Dict[str, WeatherStation] = {}
        self.max_workers = min(32, os.cpu_count() + 4)  # Optimize thread count
    
    def setup_logging(self) -> None:
        log_dir = self.base_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'parse.log'),
                logging.StreamHandler()
            ]
        )
    
    def parse_all_files(self) -> List[WeatherRecord]:
        self._parse_metadata_files()
        self._parse_extreme_weather_files()
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
            logging.info(f"Processing metadata file {i}: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                metadata = StationMetadata.from_soup(soup)
            
            # Thread-safe update of stations dictionary
            if station_number not in self.stations:
                self.stations[station_number] = WeatherStation(number=station_number, name="")
            self.stations[station_number].metadata = metadata
            
        except Exception as e:
            logging.error(f"Error processing metadata file {file_path}: {e}", exc_info=True)

    def _parse_extreme_weather_files(self) -> None:
        extreme_files = list((self.base_dir / 'raw').glob('**/extreme*.html'))
        logging.info(f"Found {len(extreme_files)} extreme weather files to process")
        
        # Process extreme weather files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(executor.map(self._process_extreme_file, enumerate(extreme_files, 1)))
    
    def _process_extreme_file(self, args):
        i, file_path = args
        try:
            station_number = file_path.parent.name
            logging.info(f"Processing extreme weather file {i}: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                extreme_data = self._parse_extreme_table(soup)
            
            # Thread-safe update of stations dictionary
            if station_number not in self.stations:
                logging.debug(f"Creating new station entry for {station_number}")
                self.stations[station_number] = WeatherStation(number=station_number, name="")
            
            # Append extreme weather data
            if hasattr(self.stations[station_number], 'extreme_weather'):
                self.stations[station_number].extreme_weather.extend(extreme_data)
            else:
                self.stations[station_number].extreme_weather = extreme_data
                
        except Exception as e:
            logging.error(f"Error processing extreme weather file {file_path}: {e}", exc_info=True)

    def _parse_extreme_table(self, soup: BeautifulSoup) -> List[ExtremeWeather]:
        """Parse extreme weather tables from HTML."""
        extreme_data = []
        tables = soup.find_all('div', {'class': 'table-responsive'})
        if not tables:
            logging.warning("No table-responsive div found in extreme weather HTML")
            return extreme_data

        # Define extreme weather types mapping
        type_mapping = {
            0: 'tmax_high',  # Highest Maximum Temperature
            1: 'tmax_low',   # Lowest Maximum Temperature
            2: 'tmin_high',  # Highest Minimum Temperature
            3: 'tmin_low',   # Lowest Minimum Temperature
            4: 'rf'          # 24 Hours Rainfall
        }

        months = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        for table_idx, table_div in enumerate(tables):
            if table_idx not in type_mapping:
                continue

            type_name = type_mapping[table_idx]
            table = table_div.find('table')
            if not table:
                continue

            rows = table.find_all('tr')[2:]  # Skip header rows
            for row_idx, row in enumerate(rows, 1):
                cells = row.find_all('td')
                if not cells:
                    continue

                try:
                    rank = int(cells[0].text.strip())
                except ValueError:
                    continue

                for month_idx, cell in enumerate(cells[1:], 0):
                    if month_idx >= len(months):
                        break
                    
                    month = months[month_idx]
                    if extreme_weather := ExtremeWeather.from_table_cell(month, rank, cell.text, type_name):
                        extreme_data.append(extreme_weather)

        return extreme_data

    def _parse_weather_files(self) -> List[WeatherRecord]:
        html_files = [f for f in self._get_html_files() 
                     if not f.name.startswith('extreme') and f.name != 'metadata.html']
        total_files = len(html_files)
        logging.info(f"Starting to process {total_files} weather files")
        records = []
        
        # Process weather files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Use a counter to track progress
            processed_count = 0
            futures = []
            
            # Submit all tasks
            for file_path in html_files:
                future = executor.submit(self._parse_file, file_path, total_files, processed_count)
                futures.append(future)
                processed_count += 1
            
            # Collect results as they complete
            for future in futures:
                result = future.result()
                if result is not None:
                    records.append(result)
        
        logging.info(f"Completed processing all {total_files} weather files")
        # Filter out None results (from errors)
        return records
    
    def _get_html_files(self) -> List[Path]:
        raw_dir = self.base_dir / 'raw'
        if not raw_dir.exists():
            raise FileNotFoundError(f"Directory '{raw_dir}' not found")
        
        html_files = []
        for subdir in raw_dir.iterdir():
            if subdir.is_dir():
                logging.info(f"Scanning directory: {subdir}")
                found_files = list(subdir.glob('*.html'))
                html_files.extend(found_files)
                logging.info(f"Found {len(found_files)} HTML files in {subdir}")
        
        logging.info(f"Total HTML files found: {len(html_files)}")
        return html_files
    
    def _parse_file(self, file_path: Path, total_files: int, file_num: int) -> Optional[WeatherRecord]:
        try:
            station_number = file_path.parent.name
            month = file_path.stem
            logging.info(f"Processing file {file_num} out of {total_files} files: {station_number}/{month}.html")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            station = WeatherStation.from_soup(soup)
            
            # Update station with metadata and extreme weather if available
            if station.number in self.stations:
                stored_station = self.stations[station.number]
                station.metadata = stored_station.metadata
                station.extreme_weather = stored_station.extreme_weather
            
            measurements = self._parse_measurements(soup)
            
            return WeatherRecord(
                month=month,
                station=station,
                measurements=measurements
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
        for row in table.find_all('tr'):
            cols = row.find_all(['th', 'td'])
            if len(cols) < 2:
                continue
                
            key = self._clean_key(cols[0].text.strip())
            if not key:
                continue
                
            if len(cols) == 3:  # 3-column format with 03UTC and 12UTC
                measurements[f"{key}_03UTC"] = self._parse_value(cols[1].text.strip())
                measurements[f"{key}_12UTC"] = self._parse_value(cols[2].text.strip())
            else:  # 2-column format
                measurements[key] = self._parse_value(cols[1].text.strip())
    
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
    
    def _save_parquet(self, records: List[WeatherRecord]) -> None:
        """Save weather data in Parquet format for efficiency and quick processing."""
        output_file = self.base_dir / 'climatological.parquet'
        try:
            # Convert records to a list of dictionaries
            data_dicts = [r.to_dict() for r in records]
            
            # Convert to pandas DataFrame first
            df = pd.DataFrame(data_dicts)
            
            # Save as Parquet
            df.to_parquet(output_file, compression='snappy')
            
            logging.info(f"Successfully saved Parquet data to {output_file}")
        except Exception as e:
            logging.error(f"Error saving Parquet file: {e}", exc_info=True)
    
    def _get_metadata_columns(self) -> List[str]:
        return [
            'station_number',
            'station_name',
            'latitude',
            'longitude',
            'elevation',
            'district',
            'state',
            'controlling_rmc',
            'controlling_mc'
        ]

    def _get_metadata_values(self, record: WeatherRecord) -> Dict[str, Any]:
        return {
            'station_number': record.station.number,
            'station_name': record.station.name,
            'latitude': record.station.metadata.latitude if record.station.metadata else float('nan'),
            'longitude': record.station.metadata.longitude if record.station.metadata else float('nan'),
            'elevation': record.station.metadata.elevation if record.station.metadata else float('nan'),
            'district': record.station.metadata.district if record.station.metadata else '',
            'state': record.station.metadata.state if record.station.metadata else '',
            'controlling_rmc': record.station.metadata.controlling_rmc if record.station.metadata else '',
            'controlling_mc': record.station.metadata.controlling_mc if record.station.metadata else ''
        }
    
    @staticmethod
    def _get_all_keys(records: List[WeatherRecord]) -> Set[str]:
        keys = set()
        for record in records:
            keys.update(record.measurements.keys())
        return keys

    def _save_csv(self, records: List[WeatherRecord]) -> None:
        """Save weather data in CSV format."""
        output_file = self.base_dir / 'climatological.csv'
        metadata_cols = self._get_metadata_columns()
        measurement_cols = sorted(self._get_all_keys(records))
        
        # Define extreme types we want to include
        extreme_types = ['tmax_high', 'tmax_low', 'tmin_high', 'tmin_low', 'rf']
        extreme_cols = []
        for ext_type in extreme_types:
            extreme_cols.append(f"{ext_type}_value")
            extreme_cols.append(f"{ext_type}_date")
        
        headers = metadata_cols + ['month'] + measurement_cols + extreme_cols

        try:
            # Create a dictionary to store extremes by station and month
            extremes_by_station_month = {}
            for record in records:
                if not record.station.extreme_weather:
                    continue
                
                station_id = record.station.number
                if station_id not in extremes_by_station_month:
                    extremes_by_station_month[station_id] = {}
                
                for extreme in record.station.extreme_weather:
                    # Only include rank 1 records
                    if extreme.rank == 1:
                        month = extreme.month
                        if month not in extremes_by_station_month[station_id]:
                            extremes_by_station_month[station_id][month] = {}
                        
                        ext_type = extreme.type
                        extremes_by_station_month[station_id][month][f"{ext_type}_value"] = extreme.value
                        extremes_by_station_month[station_id][month][f"{ext_type}_date"] = extreme.date
            
            # Create rows with both means and extremes
            rows = []
            for record in records:
                row = self._get_metadata_values(record)
                month = record.month.split('_')[0].capitalize()
                row['month'] = month
                
                # Add measurements
                for col in measurement_cols:
                    row[col] = record.measurements.get(col, float('nan'))
                
                # Add extremes if available
                station_id = record.station.number
                if station_id in extremes_by_station_month and month in extremes_by_station_month[station_id]:
                    for ext_col in extreme_cols:
                        row[ext_col] = extremes_by_station_month[station_id][month].get(ext_col, '')
                else:
                    # Initialize extreme columns with empty values
                    for ext_col in extreme_cols:
                        row[ext_col] = ''
                
                rows.append(row)
            
            # Write all rows at once
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

            logging.info(f"Successfully saved combined CSV data to {output_file}")
        except Exception as e:
            logging.error(f"Error saving combined CSV file: {e}", exc_info=True)

def main():
    base_dir = Path(__file__).parent
    
    parser = WeatherDataParser(base_dir)
    records = parser.parse_all_files()
    
    exporter = WeatherDataExporter(base_dir)
    exporter.export_data(records)
    
    logging.info("Processing completed")

if __name__ == "__main__":
    main()
