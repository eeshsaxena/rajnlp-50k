"""
Bundled query terms for Rajasthan-focused Twitter/X data collection.

These lists are used by TwitterCollector to build search queries that target
Rajasthani-Hindi code-switched content from social media.
"""

# Prominent Rajasthan politician names (current and recent)
POLITICIAN_NAMES: list[str] = [
    "Ashok Gehlot",
    "Vasundhara Raje",
    "Sachin Pilot",
    "Gajendra Singh Shekhawat",
    "Kirodi Lal Meena",
    "Arjun Ram Meghwal",
    "CP Joshi",
    "Bhajan Lal Sharma",
    "Diya Kumari",
    "Prem Chand Bairwa",
    "Rajendra Rathore",
    "Hanuman Beniwal",
    "Babulal Kharadi",
    "Govind Singh Dotasra",
    "Mahesh Joshi",
]

# Regional hashtags targeting Rajasthan content
REGIONAL_HASHTAGS: list[str] = [
    "#rajasthan",
    "#राजस्थान",
    "#jaipur",
    "#जयपुर",
    "#jodhpur",
    "#जोधपुर",
    "#udaipur",
    "#उदयपुर",
    "#kota",
    "#कोटा",
    "#ajmer",
    "#अजमेर",
    "#bikaner",
    "#बीकानेर",
    "#rajasthani",
    "#राजस्थानी",
    "#marwar",
    "#मारवाड़",
    "#mewar",
    "#मेवाड़",
    "#hadoti",
    "#shekhawati",
    "#rajpolitics",
    "#rajasthanpolitics",
]

# Documented Rajasthani slang and dialect terms
# These are common Rajasthani words/phrases used in code-switched social media text
RAJASTHANI_SLANG: list[str] = [
    "म्हारो",       # mharo — my/our (Rajasthani)
    "म्हारी",       # mhari — my/our (feminine, Rajasthani)
    "थारो",         # tharo — your (Rajasthani)
    "थारी",         # thari — your (feminine, Rajasthani)
    "कठे",          # kathe — where (Rajasthani)
    "किण",          # kin — who/which (Rajasthani)
    "बावड़ी",        # bawdi — step-well (cultural term)
    "पाणी",         # paani — water (Rajasthani variant)
    "घणो",          # ghano — very/much (Rajasthani)
    "घणी",          # ghani — very/much (feminine, Rajasthani)
    "ओळखो",         # olkho — recognize (Rajasthani)
    "बाजरो",        # bajro — pearl millet (staple crop)
    "धोरां",        # dhoran — sand dunes
    "खम्मा",        # khamma — greeting/blessing (Rajasthani)
    "सा",           # sa — honorific suffix (Rajasthani)
    "जी सा",        # ji sa — respectful address
    "राजपूत",       # rajput — community name
    "मारवाड़ी",      # marwadi — from Marwar region
    "ढोलो",         # dholo — folk song genre
    "गणगौर",        # gangaur — Rajasthani festival
    "तीज",          # teej — festival
    "पगड़ी",         # pagdi — turban
    "लहरिया",       # lahariya — traditional fabric pattern
    "बाईसा",        # baisa — respectful address for women
    "हुकम",         # hukam — command/respectful address
]
