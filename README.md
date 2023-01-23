# Grid Generator

### Getting started

Clone the repository and install the requirements:

```bash
conda env create -f environment.yml
```

### Project tree

```
📦austrian-grid
 ┣ 📂austrian_grid
 ┣ 📂data
 ┃ ┣ 📂qgis_generated_csv
 ┃ ┗ 📂raw_data
 ┣ 📂data_generator
 ┣ 📂data_generator_2
 ┣ 📂docs
 ┣ 📜environment.yml
 ┣ 📜LICENSE.md
 ┗ 📜README.md
```

where:
 * **austrian_grid**: folder containing the grid model files for the Austrian grid
 * **data**: folder containing the project data
 * **docs**: folder for project documentation (Sphinx)
 * **data_generator**: is a python package containing the code used to prepare the data for the Austrian grid
 * **data_generator_2**: is a python package containing the code used to prepare the data for a general power grid
