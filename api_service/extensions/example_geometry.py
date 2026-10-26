"""Replace analyze() with the teammate's real geometry algorithm."""


class GeometryAdapter:
    def analyze(self, context):
        return {"status": "unknown", "source": "geometry_template", "reason": "几何算法尚未接入"}


def create_adapter(options):
    return GeometryAdapter()
