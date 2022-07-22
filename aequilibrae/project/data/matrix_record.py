from os import unlink
from os.path import isfile, join
from aequilibrae.project.network.safe_class import SafeClass
from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix


class MatrixRecord(SafeClass):
    def __init__(self, data_set: dict, project):
        super().__init__(data_set, project)
        self._exists = True
        self.fldr = join(project.project_base_path, "matrices")

    def save(self):
        """Saves matrix record to the project database"""
        conn = self.connect_db()
        curr = conn.cursor()

        curr.execute("select count(*) from matrices where name=?", [self.name])
        if curr.fetchone()[0] == 0:
            data = [str(self.name), str(self.file_name), int(self.cores)]
            curr.execute("Insert into matrices (name, file_name, cores) values(?,?,?)", data)

        for key, value in self.__dict__.items():
            if key != "name" and key in self.__original__:
                v_old = self.__original__.get(key, None)
                if value != v_old and value:
                    self.__original__[key] = value
                    curr.execute(f"update matrices set '{key}'=? where name=?", [value, self.name])
        conn.commit()
        conn.close()

    def delete(self):
        """Deletes this matrix record and the underlying data from disk"""
        conn = self.connect_db()
        curr = conn.cursor()
        curr.execute("DELETE FROM matrices where name=?", [self.name])
        conn.commit()
        if isfile(join(self.fldr, self.file_name)):
            try:
                unlink(join(self.fldr, self.file_name))
            except Exception as e:
                self._logger.error(f"Could not remove matrix from disk: {e.args}")

        conn.close()
        self._exists = False

    def update_cores(self):
        """Updates this matrix record with the matrix core count in disk"""
        self.__dict__["cores"] = self.__get_cores()

    def get_data(self) -> AequilibraeMatrix:
        """Returns the actual matrix for further computation

        Returns:
            matrix (:obj:`AequilibraeMatrix`:) Matrix object
        """
        mat = AequilibraeMatrix()
        mat.load(join(self.fldr, self.file_name))
        return mat

    def __setattr__(self, instance, value) -> None:
        if instance == "name":
            value = str(value).lower()
            conn = self.connect_db()
            curr = conn.cursor()
            curr.execute("Select count(*) from matrices where LOWER(name)=?", [value])
            if sum(curr.fetchone()) > 0:
                raise ValueError("Another matrix with this name already exists")
            conn.close()
        elif instance == "file_name":
            conn = self.connect_db()
            curr = conn.cursor()
            curr.execute("Select count(*) from matrices where LOWER(file_name)=?", [str(value).lower()])
            if sum(curr.fetchone()) > 0:
                raise ValueError("There is another matrix record for this file")

        self.__dict__[instance] = value
        if instance in ["file_name", "cores"]:
            self.__dict__["cores"] = self.__get_cores()

    def __get_cores(self) -> int:
        mat = AequilibraeMatrix()
        mat.load(join(self.fldr, self.file_name))
        cores = len(mat.names)
        mat.close()
        del mat
        return cores
