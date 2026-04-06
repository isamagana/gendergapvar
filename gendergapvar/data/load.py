import pandas as pd
import numpy as np

def load_data(path=""):
    START = "1979Q1"
    END   = "2019Q2"

    def load_wide(filename, skiprows):
        df = pd.read_excel(filename, sheet_name="BLS Data Series", skiprows=skiprows)
        df = df.rename(columns={"Year": "year", "Qtr1": "Q1", "Qtr2": "Q2", "Qtr3": "Q3", "Qtr4": "Q4"})
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(float).astype(int)
        rows = []
        for _, row in df.iterrows():
            for qnum, qcol in enumerate(["Q1", "Q2", "Q3", "Q4"], start=1):
                if pd.notna(row.get(qcol)):
                    period = pd.Period(year=int(row["year"]), quarter=qnum, freq="Q")
                    rows.append({"date": period, "value": row[qcol]})
        return pd.DataFrame(rows).set_index("date")["value"].sort_index()

    gdp = pd.read_excel(f"{path}GDP.xlsx", sheet_name="Quarterly")
    gdp["date"] = pd.to_datetime(gdp["observation_date"]).dt.to_period("Q")
    gdp = gdp.set_index("date")["GDPC1"].sort_index()

    cpi = pd.read_excel(f"{path}CPI.xlsx", sheet_name="Monthly")
    cpi["date"] = pd.to_datetime(cpi["observation_date"]).dt.to_period("M")
    cpi = cpi.set_index("date")["CPIAUCSL"].sort_index()
    cpi = cpi.resample("Q").mean()

    wage_m  = load_wide(f"{path}EARNINGS_MALE.xlsx",          skiprows=19)
    wage_f  = load_wide(f"{path}EARNINGS_WOMEN.xlsx",         skiprows=19)
    lfp_m   = load_wide(f"{path}LABORFORCE_MALE.xlsx",        skiprows=12)
    lfp_f   = load_wide(f"{path}LABORFORCE_WOMEN.xlsx",       skiprows=12)
    unemp_m = load_wide(f"{path}UNEMPLOYMENTRATE_MALE.xlsx",  skiprows=12)
    unemp_f = load_wide(f"{path}UNEMPLOYMENTRATE_WOMEN.xlsx", skiprows=12)

    df = pd.concat([
        wage_m.rename("wage_m"),
        wage_f.rename("wage_f"),
        unemp_m.rename("unemp_m"),
        unemp_f.rename("unemp_f"),
        lfp_m.rename("lfp_m"),
        lfp_f.rename("lfp_f"),
        gdp.rename("gdp"),
        cpi.rename("cpi"),
    ], axis=1).loc[START:END]

    cpi_2014 = df.loc["2014Q1":"2014Q4", "cpi"].mean()
    df["wage_m"] = (df["wage_m"] / df["cpi"]) * cpi_2014
    df["wage_f"] = (df["wage_f"] / df["cpi"]) * cpi_2014

    cpi_1979q1 = df.loc["1979Q1", "cpi"]
    df["cpi"] = df["cpi"] * (29.20 / cpi_1979q1)

    df["gpg"] = (df["wage_m"] - df["wage_f"]) / df["wage_m"]

    return df