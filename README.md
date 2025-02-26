# india-climatological-stats

Dataset of climatological statistics for India. Sourced from the [Indian Meteorological Department (IMD)](https://dsp.imdpune.gov.in).

Browse the basic climatological dataset here: <https://flatgithub.com/Vonter/india-climatological-stats?filename=climatological.csv&stickyColumnName=station_name&sort=station_name>.

## Data

Dataset is available in CSV and Parquet formats:
* [climatological.csv](climatological.csv): Basic climatological statistics for each station in CSV format. Does not include all normals and includes only the highest extremes for each station.
* [climatological.parquet](climatological.parquet): All climatological statistics for each station in Parquet format. Includes all normals and extremes for each station. 

To explore or convert to other formats using a web browser, try [Konbert](https://konbert.com/viewer).

## Scripts

- [fetch.sh](fetch.sh): Fetches the raw data from the [Indian Meteorological Department (IMD)](https://dsp.imdpune.gov.in)
- [parse.py](parse.py): Parses the raw data, and generates the CSV/Parquet datasets

## License

This india-climatological-stats dataset is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/. 
Some individual contents of the database are under copyright by IMD.

You are free:

* **To share**: To copy, distribute and use the database.
* **To create**: To produce works from the database.
* **To adapt**: To modify, transform and build upon the database.

As long as you:

* **Attribute**: You must attribute any public use of the database, or works produced from the database, in the manner specified in the ODbL. For any use or redistribution of the database, or works produced from it, you must make clear to others the license of the database and keep intact any notices on the original database.
* **Share-Alike**: If you publicly use any adapted version of this database, or works produced from an adapted database, you must also offer that adapted database under the ODbL.
* **Keep open**: If you redistribute the database, or an adapted version of it, then you may use technological measures that restrict the work (such as DRM) as long as you also redistribute a version without such measures.

## Generating

Ensure you have `bash`, `curl` and `python` installed

```
# Fetch the data
bash fetch.sh

# Generate the CSV/Parquet
python parse.py
```

The fetch script sources data from Indian Meteorological Department (https://dsp.imdpune.gov.in)

## TODO

- Validations for parsing script results
- Performance optimizations for parsing script

## Credits

- [Indian Meteorological Department](https://dsp.imdpune.gov.in)
