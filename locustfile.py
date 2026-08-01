from locust import HttpUser, task, between

class YoloUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        with open("data/test_img.jpg", "rb") as f:
            self.client.post("/predict", files={"file": f})

    @task(3)
    def health(self):
        self.client.get("/health")