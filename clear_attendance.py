import sqlite3

def clear_attendance():
    conn = sqlite3.connect('users.sqlite')
    c = conn.cursor()
    c.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()
    print("Attendance records cleared.")

if __name__ == "__main__":
    clear_attendance()