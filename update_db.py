# import sqlite3

# conn = sqlite3.connect("database.db")
# c = conn.cursor()

# try:
#     c.execute("ALTER TABLE arguments ADD COLUMN highlighted_text TEXT")
#     c.execute("ALTER TABLE arguments ADD COLUMN bias_label TEXT")
#     c.execute("ALTER TABLE arguments ADD COLUMN bias_explanation TEXT")
# except:
#     pass

# conn.commit()
# conn.close()

# print("✅ Database updated successfully")


import sqlite3

conn = sqlite3.connect("database.db")
c = conn.cursor()

try:
    c.execute("ALTER TABLE arguments ADD COLUMN highlighted_text TEXT")
except:
    pass

try:
    c.execute("ALTER TABLE arguments ADD COLUMN bias_label TEXT")
except:
    pass

try:
    c.execute("ALTER TABLE arguments ADD COLUMN bias_explanation TEXT")
except:
    pass

# ✅ ADD THIS
try:
    c.execute("ALTER TABLE arguments ADD COLUMN username TEXT DEFAULT 'Anonymous'")
except:
    pass

conn.commit()
conn.close()

print("✅ Database updated successfully")