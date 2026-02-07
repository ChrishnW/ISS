import time

class Cooldown:
    def __init__(self, seconds):
        self.seconds = seconds
        self.last_seen = {}

    def allowed(self, person_id):
        now = time.time()
        last = self.last_seen.get(person_id, 0)

        if now - last > self.seconds:
            self.last_seen[person_id] = now
            return True

        return False
