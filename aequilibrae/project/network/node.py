from .safe_class import SafeClass
from aequilibrae.project.database_connection import database_connection
from aequilibrae import logger


class Node(SafeClass):
    """A Node object represents a single record in the *nodes* table

    ::

        from aequilibrae import Project

        proj = Project()
        proj.open('path/to/project/folder')

        all_nodes = proj.network.nodes

        # We can just get one link in specific
        node1 = all_nodes.get(7890)

        # We can find out which fields exist for the links
        which_fields_do_we_have = node1.data_fields()

        # And edit each one like this
        node1.comment = 'This node is important'

        # We can just save the node
        node1.save()
        """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.__fields = list(dataset.keys())

    def save(self):
        """Saves node to database"""
        conn = database_connection()
        curr = conn.cursor()

        data = []

        if self.node_id != self.__original__['node_id']:
            raise ValueError('One cannot change the node_id')

        txts = []
        for key, val in self.__dict__.items():
            if key not in self.__original__:
                continue
            if val != self.__original__[key]:
                if key == 'geometry' and val is not None:
                    data.append(val.wkb)
                    txts.append('geometry=GeomFromWKB(?, 4326)')
                else:
                    data.append(val)
                    txts.append(f'"{key}"=?')

        if not data:
            logger.warn(f'Nothing to update for node {self.node_id}')
            return

        txts = ','.join(txts) + ' where node_id=?'
        data.append(self.node_id)
        sql = f'Update Nodes set {txts}'

        curr.execute(sql, data)
        conn.commit()
        conn.close()
        self.__new = False

    def data_fields(self) -> list:
        """lists all data fields for the node, as available in the database

        Returns:
            *data fields* (:obj:`list`): list of all fields available for editing
        """

        return list(self.__original__.keys())

    def renumber(self, new_id: int):
        """Renumbers the node in the network

        Raises a warning if another node already exists with this node_id

        Args:
            *new_id* (:obj:`int`): New node_id
        """

        if new_id == self.node_id:
            raise ValueError('This is already the node number')

        conn = database_connection()
        curr = conn.cursor()

        curr.execute('BEGIN;')
        curr.execute('Update Nodes set node_id=? where node_id=?', [new_id, self.node_id])
        curr.execute('Update Links set a_node=? where a_node=?', [new_id, self.node_id])
        curr.execute('Update Links set b_node=? where b_node=?', [new_id, self.node_id])
        curr.execute('COMMIT;')
        conn.close()
        self.__dict__['node_id'] = new_id
        self.__original__['node_id'] = new_id

    def __setattr__(self, instance, value) -> None:
        if instance not in self.__dict__ and instance[:1] != "_":
            raise AttributeError(f'"{instance}" is not a valid attribute for a node')
        elif instance == 'node_id':
            raise AttributeError('Setting node_id is not allowed')
        elif instance == 'link_types':
            raise AttributeError('Setting link_types is not allowed')
        elif instance == 'modes':
            raise AttributeError('Setting modes is not allowed')
        elif instance == 'is_centroid':
            if value not in [0, 1]:
                raise ValueError('The is_centroid must be either 1 or 0')
        self.__dict__[instance] = value
