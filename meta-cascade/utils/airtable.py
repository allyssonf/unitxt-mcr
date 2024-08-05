import os
import time
import datetime
import socket

from enum import Enum
from pyairtable.orm import Model, fields as F

class Result(Enum):
    started = 'InProgress'
    success = 'Done'
    failure = 'Failed'

class Datasets(Model):
    name = F.TextField('Name')

    class Meta:
        base_id = "appYtoQ3Hs6pU2USZ"
        table_name = "Datasets"
        api_key = os.getenv('AIRTABLE_API_KEY')

class Models(Model):
    name = F.TextField('Name')

    class Meta:
        base_id = "appYtoQ3Hs6pU2USZ"
        table_name = "Models"
        api_key = os.getenv('AIRTABLE_API_KEY')

class DGLogs(Model):
    model = F.LinkField('Model', Models)
    dataset = F.LinkField('Dataset', Datasets)
    run_name = F.TextField('Run')
    host = F.TextField('Host')
    status = F.SelectField('Status')
    start = F.TextField('Start')
    end = F.TextField('End')
    duration = F.TextField('Duration')
    notes = F.TextField('Notes')

    class Meta:
        base_id = "appYtoQ3Hs6pU2USZ"
        table_name = "DG Logs"
        api_key = os.getenv('AIRTABLE_API_KEY')


class AirTableLogger:
    record_instance: DGLogs | None
    start: float

    def __init__(self):
        self.record_instance = None
        self.start = 0

    def get_eval_model(self, model_name: str) -> Models | None:
        for evlmdl in Models.all():
            if evlmdl.name.lower() == model_name.lower():
                return evlmdl
        return None
    
    def get_eval_dataset(self, dataset_name: str) -> Datasets | None:
        for evldtst in Datasets.all():
            if evldtst.name.lower() == dataset_name.lower():
                return evldtst
        return None

    def log_start(self, run_name: str, model_name: str, dataset_name: str) -> None:
        self.start = time.perf_counter()

        eval_model = self.get_eval_model(model_name)

        assert eval_model is not None

        eval_dataset = self.get_eval_dataset(dataset_name)

        assert eval_dataset is not None

        # Create Log Record 
        new_log = DGLogs(
            model = [eval_model],
            dataset = [eval_dataset],
            run_name = run_name,
            host = socket.gethostname(),
            status = Result.started.value,
            start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            end = '',
            duration = '',
            notes = '',
        )

        if new_log.save():
            self.record_instance = new_log

    def log_end(self, status: Result, message: str | None = None) -> None:
        self.record_instance.end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        end_time = time.perf_counter()
        duration = str(datetime.timedelta(seconds=end_time-self.start))

        self.record_instance.duration = duration
        self.record_instance.notes = message or ''
        self.record_instance.status = status.value

        self.record_instance.save()
