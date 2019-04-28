class Trip:
    def __init__(self):
        # Some characteristics are not needed because they will be inherited from a parent class
        # self.route_id

        # Others are needed
        self.service_id = None
        self.id = None
        self.head_sign = None
        self.short_name = None
        self.direction_id = None
        self.block_id = None
        self.shape_id = None
        self.wheelchair_accessible = None
        self.bikes_allowed = None
