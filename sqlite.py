import sqlite3

conn = sqlite3.connect('users.sqlite')
c = conn.cursor()

c.execute('''CREATE TABLE users (username text, password text, user_type text)''')

c.execute("INSERT INTO users VALUES ('staff_username','staff_password','Staff')")
c.execute("INSERT INTO users VALUES ('admin_username','admin_password','Admin')")

c.execute('''CREATE TABLE attendance (Name text, roll_no text, Time text, Date text, user text, Period text)''')

conn.commit()
conn.close()
