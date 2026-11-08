import sys
import zipfile
import csv
import httpx

MAX_USERS = 200
MAX_POSTS = 2000
API_URL = "http://127.0.0.1:9999"

def seed_database(zip_path: str):
    print(f"Reading MovieLens data from {zip_path}...")
    
    with httpx.Client(timeout=None) as client:
        with zipfile.ZipFile(zip_path, 'r') as z:
            movies_filename = next(name for name in z.namelist() if name.endswith('movies.csv'))
            ratings_filename = next(name for name in z.namelist() if name.endswith('ratings.csv'))

            posts_added = 0
            valid_movie_ids = set()
            with z.open(movies_filename) as f:
                reader = csv.DictReader(f.read().decode('utf-8').splitlines())
                for row in reader:
                    if posts_added >= MAX_POSTS:
                        break
                    movie_id = int(row['movieId'])
                    title = row['title']
                    
                    client.post(f"{API_URL}/posts", json={"user_id": 9999, "content": title})
                    valid_movie_ids.add(movie_id)
                    posts_added += 1

            print(f"Added {posts_added} movies as posts.")

            users_added = set()
            likes_added = 0
            
            with z.open(ratings_filename) as f:
                reader = csv.DictReader(f.read().decode('utf-8').splitlines())
                for row in reader:
                    user_id = int(row['userId'])
                    movie_id = int(row['movieId'])
                    rating = float(row['rating'])
                    timestamp = float(row['timestamp'])

                    if len(users_added) >= MAX_USERS and user_id not in users_added:
                        continue

                    if user_id not in users_added:
                        client.post(f"{API_URL}/users", json={"username": f"ML_User_{user_id}"})
                        users_added.add(user_id)

                    if rating >= 4.0 and movie_id in valid_movie_ids:
                        client.post(
                            f"{API_URL}/likes", 
                            json={
                                "user_id": user_id, 
                                "post_id": movie_id, 
                                "created_at": timestamp
                            }
                        )
                        likes_added += 1

            print(f"Added {len(users_added)} users and {likes_added} likes.")
            print("Seeding complete! Data is deterministic and bounded.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python seed_movielens.py <path_to_ml-latest-small.zip>")
        sys.exit(1)
    
    seed_database(sys.argv[1])