# data/crisis_library.py

CRISIS_LIBRARY = {
    "LTCM_1998": {
        "name": "LTCM Collapse",
        "short": "LTCM 1998",
        "description": (
            "Long-Term Capital Management, a highly leveraged hedge fund, collapsed after "
            "Russian debt default. Required a Fed-orchestrated bailout. Key fingerprint: "
            "basis blowout in theoretically related instruments, liquidity withdrawal from "
            "obscure spreads before equity markets noticed."
        ),
        "stress_start": "1998-06-01",
        "stress_end":   "1998-09-01",
        "peak_date":    "1998-10-08",
        "color":        "#E74C3C",
        "key_signature": "Basis blowouts + liquidity fragmentation"
    },
    "DOTCOM_2000": {
        "name": "Dot-Com Bust",
        "short": "Dot-Com 2000",
        "description": (
            "Technology bubble collapse. NASDAQ fell 78% peak to trough. Key fingerprint: "
            "intra-equity sector divergence (tech vs value), VIX term structure flattening, "
            "without the broad credit market stress seen in GFC."
        ),
        "stress_start": "2000-01-01",
        "stress_end":   "2000-03-10",
        "peak_date":    "2000-03-10",
        "color":        "#E67E22",
        "key_signature": "Intra-equity sector divergence + vol term structure"
    },
    "GFC_2008": {
        "name": "Global Financial Crisis",
        "short": "GFC 2008",
        "description": (
            "Subprime mortgage collapse triggered global banking crisis. Key fingerprint: "
            "TED spread explosion, commercial paper market seizure, everything correlated to 1, "
            "funding markets frozen before equity indices reflected the stress."
        ),
        "stress_start": "2007-08-01",
        "stress_end":   "2008-09-15",
        "peak_date":    "2008-10-10",
        "color":        "#8E44AD",
        "key_signature": "Funding market seizure + universal correlation spike"
    },
    "FLASH_CRASH_2010": {
        "name": "Flash Crash",
        "short": "Flash Crash 2010",
        "description": (
            "May 6, 2010: Dow Jones fell nearly 1000 points in minutes then recovered. "
            "Key fingerprint: pure liquidity microstructure stress without credit signal, "
            "very short duration. Demonstrates positioning-driven stress."
        ),
        "stress_start": "2010-04-15",
        "stress_end":   "2010-05-06",
        "peak_date":    "2010-05-06",
        "color":        "#27AE60",
        "key_signature": "Microstructure-only liquidity collapse, no credit signal"
    },
    "EUROZONE_2011": {
        "name": "Eurozone Debt Crisis",
        "short": "Eurozone 2011",
        "description": (
            "Greek, Italian, Spanish sovereign debt crisis threatened euro breakup. "
            "Key fingerprint: geographic contagion pattern in sovereign credit, EUR/USD stress, "
            "financials leading equity drawdown."
        ),
        "stress_start": "2011-06-01",
        "stress_end":   "2011-08-08",
        "peak_date":    "2011-09-23",
        "color":        "#2980B9",
        "key_signature": "Sovereign credit + financials leading + EUR stress"
    },
    "TAPER_TANTRUM_2013": {
        "name": "Taper Tantrum",
        "short": "Taper Tantrum 2013",
        "description": (
            "Fed signaled QE tapering; bond markets sold off violently. "
            "Key fingerprint: rate-driven cross-asset stress, EM currency selloff, "
            "bond-equity correlation flip, yield curve steepening velocity."
        ),
        "stress_start": "2013-05-01",
        "stress_end":   "2013-06-25",
        "peak_date":    "2013-06-25",
        "color":        "#16A085",
        "key_signature": "Duration selloff + EM outflows + yield curve velocity"
    },
    "CHINA_OIL_2015": {
        "name": "China/Oil Shock",
        "short": "China/Oil 2015",
        "description": (
            "Chinese growth fears + oil collapse triggered global equity selloff. "
            "Key fingerprint: commodity-financial system coupling, EM stress, "
            "energy sector leading, oil-dollar relationship breaking down."
        ),
        "stress_start": "2015-06-01",
        "stress_end":   "2015-08-25",
        "peak_date":    "2015-08-25",
        "color":        "#D35400",
        "key_signature": "Commodity-financial coupling + EM selloff"
    },
    "VOL_SHOCK_2018": {
        "name": "Volmageddon",
        "short": "Volmageddon 2018",
        "description": (
            "February 2018: Short volatility strategies (VIX ETPs) exploded, "
            "triggering forced unwinds. Key fingerprint: VIX futures basis blowout, "
            "positioning-driven without fundamental credit stress, very fast recovery."
        ),
        "stress_start": "2018-01-15",
        "stress_end":   "2018-02-05",
        "peak_date":    "2018-02-05",
        "color":        "#C0392B",
        "key_signature": "VIX futures basis + short-vol unwind + no credit"
    },
    "COVID_2020": {
        "name": "COVID Market Crash",
        "short": "COVID 2020",
        "description": (
            "March 2020: Fastest 30% market crash in history. Key fingerprint: "
            "everything selling simultaneously including gold (margin calls), "
            "correlations going to 1 across ALL assets, dollar surging, "
            "funding markets seizing within days."
        ),
        "stress_start": "2020-02-01",
        "stress_end":   "2020-03-23",
        "peak_date":    "2020-03-23",
        "color":        "#1ABC9C",
        "key_signature": "Universal correlation spike + margin call fingerprint"
    },
    "SVB_2023": {
        "name": "SVB Banking Crisis",
        "short": "SVB 2023",
        "description": (
            "Silicon Valley Bank collapse triggered regional banking crisis. "
            "Key fingerprint: regional bank CDS spike, financials sector stress, "
            "rate sensitivity stress without broad credit market contagion. "
            "Funding-specific rather than credit-market-wide."
        ),
        "stress_start": "2023-02-01",
        "stress_end":   "2023-03-10",
        "peak_date":    "2023-03-10",
        "color":        "#9B59B6",
        "key_signature": "Financials-specific + rate sensitivity + contained credit"
    }
}
