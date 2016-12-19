from locust import HttpLocust, TaskSet


def qtype_classify(l):
    l.client.get("/qtype?q=How do the Nazis justify the killings of jews in the Holocaust")


def index(l):
    l.client.get("/")


class UserBehavior(TaskSet):
    tasks = {qtype_classify: 1}

    # def on_start(self):
    #     index(self)


class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    min_wait = 0
    max_wait = 0
