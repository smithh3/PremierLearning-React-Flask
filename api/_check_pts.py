import sqlite3
conn = sqlite3.connect("../database/fantasy.db")
c = conn.cursor()
print("avg/max/min future points", c.execute("select avg(points), max(points), min(points) from future_matches").fetchone())
print("top 10:")
for r in c.execute("""
  select p.first_name, p.surname, f.points, f.minutes, f.opponent, f.gameweek
  from future_matches f join players p on p.id=f.player_id
  order by f.points desc limit 10
"""):
    print(r)
print("Haaland:")
for r in c.execute("""
  select f.gameweek, f.opponent, f.points, f.minutes
  from future_matches f join players p on p.id=f.player_id
  where p.surname like '%Haaland%'
  order by f.gameweek
"""):
    print(r)
print("y train sample if exists")
