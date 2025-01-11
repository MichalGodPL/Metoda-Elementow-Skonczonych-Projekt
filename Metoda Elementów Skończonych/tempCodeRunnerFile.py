    def Jakobian(self, dN_dxi):

        J = np.dot(dN_dxi, self.nodes)

        return J