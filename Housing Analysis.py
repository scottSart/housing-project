import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1. FILE PATHS

PERMIT_DATA_PATH = r'C:\Users\Sarth\OneDrive\JHU\HomeEconomics\Project 2  - Relationship between Rent prices and home permits\Consolidated Dataset_v2.xlsx'
RENTAL_DATA_PATH = r'C:\Users\Sarth\OneDrive\JHU\HomeEconomics\Project 2  - Relationship between Rent prices and home permits\Consolidated Dataset_v2.xlsx'
APARTMENT_LIST_PATH = r'C:\Users\Sarth\OneDrive\JHU\HomeEconomics\Project 2  - Relationship between Rent prices and home permits\Apartment_List_Rent_Estimates_2024_09.csv'
APARTMENT_LIST_2025_PATH = r'C:\Users\Sarth\OneDrive\JHU\HomeEconomics\Project 2  - Relationship between Rent prices and home permits\Apartment_List_Rent_Estimates_2025_01.csv'
POPULATION_DATA_PATH = r'C:\Users\Sarth\Downloads\population.xlsx'

# 2. HELPER FUNCTIONS
def cagr(start_value, end_value, num_periods):
    """
    Compound Annual Growth Rate.
    If num_periods = number of intervals, then CAGR = (end / start)^(1/(num_periods)) - 1.
    """
    return (end_value / start_value) ** (1 / num_periods) - 1

# 3. LOAD DATASETS
# Permit data (Residential_Construction_Permit sheet)
permit_data = pd.read_excel(PERMIT_DATA_PATH, sheet_name="Residential_Construction_Permit")
permit_data['location_name'] = permit_data['NAME'] + ", " + permit_data['STUSAB']
permit_data.rename(columns={'NAME': 'County Name', 'STATE_NAME': 'State', 'GEOID': 'Geo ID'}, inplace=True)

# Rental data (Median Rent sheet)
rental_data = pd.read_excel(RENTAL_DATA_PATH, sheet_name="Median Rent")

# Apartment List data (overall rent, multiple files)
apartment_list_1 = pd.read_csv(APARTMENT_LIST_PATH)
apartment_list_2 = pd.read_csv(APARTMENT_LIST_2025_PATH)

# Population data (sheet = "Main")
popn = pd.read_excel(POPULATION_DATA_PATH, sheet_name="Main")
popn.rename(columns={'City_st': 'location_name'}, inplace=True)

# 4. APARTMENT LIST EXAMPLE ANALYSIS (Bubble Scatter Plots)

apartment_list_all = apartment_list_1[apartment_list_1['bed_size'] == 'overall']

# Separate out by County / City
apt_list_county = apartment_list_all[apartment_list_all['location_type'] == 'County'].copy()
apt_list_city   = apartment_list_all[apartment_list_all['location_type'] == 'City'].copy()

year_columns = ["2024_05", "2023_05", "2022_05", "2021_05", "2020_05", "2019_05", "2018_05", "2017_05"]
red_columns = ['location_name', 'location_type', 'location_fips_code', 'state', 'county', 'metro',
               'population'] + year_columns

apt_list_city_red = apt_list_city[red_columns].copy()

# Calculate yoy % changes
# do a single loop so we match your original snippet
pct_change_cols = {}
num_years = len(year_columns)
for i in range(num_years - 1):
    col_curr = year_columns[i]
    col_next = year_columns[i + 1]
    pct_name = f"{col_curr}_v_{col_next}"
    apt_list_city_red[pct_name] = ((apt_list_city_red[col_curr] - apt_list_city_red[col_next])
                                   / apt_list_city_red[col_next]) * 100

apt_list_city_red["2020_2022"] = cagr(
    apt_list_city_red["2020_05"], apt_list_city_red["2022_05"], 2
) * 100

# Drop original rent columns for clarity
apt_list_pct_change = apt_list_city_red.drop(columns=year_columns).dropna()

plt.figure(figsize=(7, 5))
plt.scatter(
    apt_list_pct_change["2020_2022"],
    apt_list_pct_change["2024_05_v_2023_05"],
    s=apt_list_pct_change["population"] / 30000,
    alpha=0.7
)
plt.xlabel("Rent CAGR (2020–2022)")
plt.ylabel("Rent % Change (2023->2024)")
plt.title("Bubble Scatter: Rent Change 2020–22 vs. Rent Change 2023–24")

# Annotate Austin
plt.annotate("Austin, TX", xy=(27.265625, -7.416719),
             xytext=(30, -10),
             arrowprops=dict(arrowstyle='->'),
             fontsize=9)
plt.annotate("New York City, NY", xy=(17.026749, 2.73913),
             xytext=(20, 5),
             arrowprops=dict(arrowstyle='->'),
             fontsize=9)

plt.tight_layout()
plt.show()

# 5. ADD POPULATION DATA MERGE & EXTRA BUBBLE PLOTS

data_w_pop = pd.merge(apt_list_pct_change, popn, on='location_name', how='left')
data_w_pop['pop_22v23'] = ((data_w_pop[2023] / data_w_pop[2022]) - 1) * 100
data_w_pop['pop_20v22'] = cagr(data_w_pop[2020], data_w_pop[2022], 2) * 100

plt.figure(figsize=(7,5))
plt.scatter(
    data_w_pop["pop_20v22"],
    data_w_pop["2020_2022"],
    s=data_w_pop["population"] / 20000,
    alpha=0.5
)
plt.xlabel("Population CAGR (2020–2022)")
plt.ylabel("Rent CAGR (2020–2022)")
plt.title("Rent vs. Population Change (2020–22)")
plt.tight_layout()
plt.show()

# 6. PERMIT DATA WRANGLING (Pivot, Cumulative, YOY)
permit_data_red = permit_data[['Geo ID', 'location_name', 'Year', 'Multi Family Permits']].copy()
permit_data_red.dropna(subset=['Multi Family Permits'], inplace=True)

permit_data_pivot = permit_data_red.pivot(
    index=['Geo ID', 'location_name'],
    columns='Year',
    values='Multi Family Permits'
).reset_index()

# Ensure numeric
numeric_year_cols = permit_data_pivot.columns[2:]
for col in numeric_year_cols:
    permit_data_pivot[col] = pd.to_numeric(permit_data_pivot[col], errors='coerce')

# Cumulative sum across columns
permit_data_pivot[numeric_year_cols] = permit_data_pivot[numeric_year_cols].cumsum(axis=1)

# Convert cumulative sums to yoy % changes
yoy_df = permit_data_pivot.copy()
yoy_cols = yoy_df.columns[2:]
yoy_df[yoy_cols] = yoy_df[yoy_cols].pct_change(axis=1) * 100

# 7. RENT & POPULATION (Median Rent sheet)
rental_data_red = rental_data[[
    'Geo ID', 'County Name', 'State', 'Estimate!!GROSS RENT!!Median (dollars)', 'Year', 'Population'
]].rename(columns={'Estimate!!GROSS RENT!!Median (dollars)': 'med_rent'})
rental_data_long = rental_data_red.pivot(
    index=['Geo ID', 'County Name', 'State'],
    columns='Year',
    values=['med_rent', 'Population']
)
rental_data_long.columns = [f"{lvl1}_{lvl2}" for lvl1, lvl2 in rental_data_long.columns]
rental_data_long.reset_index(inplace=True)

# Build yoy % changes for 2010..2023
rent_change_dict = {}
pop_change_dict = {}
year_range = range(2010, 2024)
for s, e in zip(year_range, year_range[1:]):
    rs, re = f"med_rent_{s}", f"med_rent_{e}"
    ps, pe = f"Population_{s}", f"Population_{e}"
    if rs in rental_data_long and re in rental_data_long:
        rent_change_dict[f"{rs}_v_{re}"] = ((rental_data_long[re] - rental_data_long[rs])
                                            / rental_data_long[rs]) * 100
    if ps in rental_data_long and pe in rental_data_long:
        pop_change_dict[f"{ps}_v_{pe}"] = ((rental_data_long[pe] - rental_data_long[ps])
                                           / rental_data_long[ps]) * 100

yoy_rent_df = pd.DataFrame(rent_change_dict)
yoy_pop_df = pd.DataFrame(pop_change_dict)

# Attach location info
yoy_rent_df = pd.concat([rental_data_long[['Geo ID', 'County Name', 'State']], yoy_rent_df], axis=1)
yoy_pop_df  = pd.concat([rental_data_long[['Geo ID', 'County Name', 'State']], yoy_pop_df], axis=1)

combined_df = pd.merge(yoy_pop_df, yoy_rent_df, on=['Geo ID', 'County Name', 'State'])

# 8. MERGE PERMIT YOY WITH RENT & POP YOY
final_permit_df = yoy_df.copy()
final_permit_df.columns = final_permit_df.columns.map(str)

df_pop_rent_permit = pd.merge(combined_df, final_permit_df, on='Geo ID', how='inner')

# 9. CORRELATION BAR CHARTS (Pop vs Rent, Permits vs Rent)
year_pairs = [
    (2010, 2011), (2011, 2012), (2012, 2013), (2013, 2014),
    (2014, 2015), (2015, 2016), (2016, 2017), (2017, 2018),
    (2018, 2019), (2019, 2021), (2021, 2022), (2022, 2023)
]

pop_correlations = []
permit_correlations = []
year_labels = []

for (start, end) in year_pairs:
    pop_col = f"Population_{start}_v_Population_{end}"
    rent_col = f"med_rent_{start}_v_med_rent_{end}"
    permit_col = str(start)  # yoy col from final_permit_df

    needed = [pop_col, rent_col, permit_col]
    if not all(col in df_pop_rent_permit.columns for col in needed):
        continue

    df_temp = df_pop_rent_permit[needed].replace([np.inf, -np.inf], np.nan).dropna()
    if df_temp.empty:
        continue

    pop_correlations.append(df_temp[pop_col].corr(df_temp[rent_col]))
    permit_correlations.append(df_temp[permit_col].corr(df_temp[rent_col]))
    year_labels.append(f"{start}-{end}")

x_vals = np.arange(len(year_labels))
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x_vals - bar_width/2, pop_correlations, bar_width, label='Population vs. Rent')
plt.bar(x_vals + bar_width/2, permit_correlations, bar_width, label='Permits vs. Rent')
plt.ylabel("Correlation")
plt.title("Year-over-Year Correlation with Rent Changes")
plt.xticks(x_vals, year_labels, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 10. MULTIPLE REGRESSION COEFFICIENTS BY YEAR
coef_pop = []
coef_permit = []
labels = []

for (start, end) in year_pairs:
    pop_col = f"Population_{start}_v_Population_{end}"
    rent_col = f"med_rent_{start}_v_med_rent_{end}"
    permit_col = str(start)

    if not all(col in df_pop_rent_permit.columns for col in [pop_col, rent_col, permit_col]):
        continue

    df_tmp = df_pop_rent_permit[[pop_col, rent_col, permit_col]].dropna()
    if df_tmp.empty:
        continue

    X = df_tmp[[pop_col, permit_col]]
    y = df_tmp[rent_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    if pop_col in model.params and permit_col in model.params:
        coef_pop.append(model.params[pop_col])
        coef_permit.append(model.params[permit_col])
        labels.append(f"{start}-{end}")

plt.figure(figsize=(10, 6))
plt.plot(labels, coef_pop, marker='o', label='Coefficient - Population')
plt.plot(labels, coef_permit, marker='s', label='Coefficient - Permits')
plt.title("Multiple Regression Coefficients by Year")
plt.xlabel("Year Ranges")
plt.ylabel("Coefficient")
plt.xticks(rotation=45, ha='right')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 11. SCATTER PLOTS (Population vs. Rent, Permits vs. Rent)
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))
axes = axes.flatten()

for i, (start, end) in enumerate(year_pairs):
    if i >= len(axes):
        break

    pop_col   = f"Population_{start}_v_Population_{end}"
    rent_col  = f"med_rent_{start}_v_med_rent_{end}"
    permit_col= str(start)

    needed = [pop_col, rent_col, permit_col]
    if not all(col in df_pop_rent_permit.columns for col in needed):
        axes[i].axis('off')
        continue

    df_tmp = df_pop_rent_permit[needed].dropna()
    if df_tmp.empty:
        axes[i].axis('off')
        continue

    # Pop vs Rent
    axes[i].scatter(df_tmp[pop_col], df_tmp[rent_col], alpha=0.3, color='blue', label='Pop vs. Rent')
    slope, intercept = np.polyfit(df_tmp[pop_col], df_tmp[rent_col], 1)
    axes[i].plot(df_tmp[pop_col], slope*df_tmp[pop_col] + intercept, color='blue')

    # Permits vs Rent
    axes[i].scatter(df_tmp[permit_col], df_tmp[rent_col], alpha=0.2, color='red', label='Permits vs. Rent')
    slope_p, intercept_p = np.polyfit(df_tmp[permit_col], df_tmp[rent_col], 1)
    axes[i].plot(df_tmp[permit_col], slope_p*df_tmp[permit_col] + intercept_p, color='red')

    axes[i].set_title(f"{start}-{end}")
    axes[i].legend()

plt.tight_layout()
plt.show()

# 12. ONE MODEL ON ALL YEARS + HYPOTHETICAL SCENARIO
all_rows = []
for (start, end) in year_pairs:
    pop_col = f"Population_{start}_v_Population_{end}"
    rent_col= f"med_rent_{start}_v_med_rent_{end}"
    permit_col = str(start)

    if not all(col in df_pop_rent_permit.columns for col in [pop_col, rent_col, permit_col]):
        continue

    df_t = df_pop_rent_permit[[pop_col, rent_col, permit_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if df_t.empty:
        continue

    df_t = df_t.rename(columns={
        pop_col: "pop_change",
        rent_col: "rent_change",
        permit_col: "permit_change"
    })
    df_t["year"] = start
    all_rows.append(df_t)

df_yoy = pd.concat(all_rows, ignore_index=True).dropna(subset=["pop_change", "rent_change", "permit_change"])

X_full = sm.add_constant(df_yoy[["pop_change", "permit_change"]])
y_full = df_yoy["rent_change"]
model_full = sm.OLS(y_full, X_full).fit()
print(model_full.summary())

# Hypothetical scenario: Zero out permit_change for year >= 2011
df_hypo = df_yoy.copy()
df_hypo.loc[df_hypo["year"] >= 2011, "permit_change"] = 0.0

X_hypo = sm.add_constant(df_hypo[["pop_change", "permit_change"]])
df_hypo["rent_change_hypo"] = model_full.predict(X_hypo)

# Compare average actual vs. hypothetical
actual_avg = df_yoy.groupby("year", as_index=False)["rent_change"].mean()
hypo_avg   = df_hypo.groupby("year", as_index=False)["rent_change_hypo"].mean()
df_compare = pd.merge(actual_avg, hypo_avg, on="year", how="inner").sort_values("year")

# Build a rent index (base=100)
rent_index_actual = []
rent_index_hypo   = []
curr_act = 100.0
curr_hyp = 100.0

for _, row in df_compare.iterrows():
    yr   = row["year"]
    yoyA = row["rent_change"] / 100.0
    yoyH = row["rent_change_hypo"] / 100.0

    curr_act *= (1.0 + yoyA)
    curr_hyp *= (1.0 + yoyH)

    rent_index_actual.append((yr + 1, curr_act))
    rent_index_hypo.append((yr + 1, curr_hyp))

df_index_actual = pd.DataFrame(rent_index_actual, columns=["year", "rent_index_actual"])
df_index_hypo   = pd.DataFrame(rent_index_hypo, columns=["year", "rent_index_hypo"])
df_plot = pd.merge(df_index_actual, df_index_hypo, on="year", how="outer").sort_values("year")

plt.figure(figsize=(8, 5))
plt.plot(df_plot["year"], df_plot["rent_index_actual"], marker='o', label='Actual Permits')
plt.plot(df_plot["year"], df_plot["rent_index_hypo"], marker='s', label='No Permits After 2011')
plt.title("Hypothetical: No Permits Approved After 2011")
plt.xlabel("Year")
plt.ylabel("Rent Index (Base=100)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


