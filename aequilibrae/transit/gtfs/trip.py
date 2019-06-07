class Trip:
    def __init__(self):
        # Some characteristics are not needed because they will be inherited from a parent class
        # self.route_id

        # Others are needed
        self.service_id: str = None
        self.id: str = None
        self.head_sign: str = None
        self.short_name: str = None
        self.direction_id: str = None
        self.block_id: str = None
        self.shape_id: str = None
        self.wheelchair_accessible: int = None
        self.bikes_allowed: int = None
