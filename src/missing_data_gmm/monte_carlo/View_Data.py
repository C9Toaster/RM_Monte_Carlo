import pandas as pd

from missing_data_gmm.config import DATA_CATALOGS

# Example: Viewing simulation results for a specific design and missingness option
design = 1
missingness = "MCAR"

# Access the DataCatalog for the specified design and missingness option
data_catalog = DATA_CATALOGS["simulation"]

# Load the data
data_path = data_catalog[f"MC_RESULTS_{design}_{missingness}"].path
data = pd.read_csv(data_path)

# Display the data
print(data.head(20))
