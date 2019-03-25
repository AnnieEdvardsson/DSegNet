from framework_component import FrameworkComponent

class KerasModelTemplate(FrameworkComponent):
    def create_model(self):
        raise NotImplementedError()
