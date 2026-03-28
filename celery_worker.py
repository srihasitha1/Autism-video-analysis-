from celery import Celery

celery = Celery(
    'autisense',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery.task
def test_task(x, y):
    return x + y
