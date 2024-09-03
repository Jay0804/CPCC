from skmultiflow.data import DataStream
import pandas as pd
import numpy as np
import csv

from legacy.str_DMGT import DMGT_strategy
from OAL_strategies.str_MTSGQS import MTSGQS_strategy
from OAL_strategies.str_DSA_AI import DSA_AI_strategy
from OAL_strategies.str_US_fix import US_fix_strategy
from OAL_strategies.str_US_var import US_var_strategy
from OAL_strategies.str_CogDQS import CogDQS_strategy
from OAL_strategies.str_ROALE_DI import ROALE_DI_strategy
from OAL_strategies.str_RS import RS_strategy
from OAL_strategies.str_USGQS import USGQS_strategy

from classifier.clf_BLS import BLS
from classifier.clf_BLS_G import BLS_G
from classifier.clf_BLS_pseudo import BLS_pseudo
from classifier.clf_SRP import SRP
from classifier.clf_DES import DES_ICD
from skmultiflow.bayes import NaiveBayes

from OSSL_classifier.clf_OSSBLS import OSSBLS
from OSSL_classifier.clf_OSSBLS_pseudo import OSSBLS_pseudo
from OSSL_classifier.clf_OSSBLS_ori import OSSBLS_ori
from OSSL_classifier.clf_ODSSBLS import ODSSBLS
from OSSL_classifier.clf_ISSBLS import ISSBLS
from OSSL_classifier.clf_ISSBLS_pseudo import ISSBLS_pseudo
from OSSL_classifier.clf_SOSELM import SOSELM
from OSSL_classifier.clf_tri_VFDT import tri_VFDT
from OSSL_classifier.clf_OSSEL import OSSEL
from OSSL_classifier.clf_OMISRVFL import OMIS_RVFL
from OSSL_classifier.clf_E_BLS import E_BLS
from OSSL_classifier.clf_ODSSBLS import ODSSBLS
from OSSL_classifier.clf_ODSSBLS_cluster_replace import ODSSBLS_cluster
from OSSL_classifier.clf_OMISRVFL_new import OMIS_RVFL_new
from OSSL_classifier.clf_MPPOSRVFL import MPPOS_RVFL
from OSSL_classifier.clf_CPSSDS import CPSSDS
from sklearn.neighbors import KNeighborsClassifier
from classifier.clf_ARF import ARF
from sklearn import svm

class para_init:

    def __init__(self, n_class, n_anchor):
        self.n_class = n_class
        self.n_anchor = n_anchor
        self.reg = 0.00001
        self.gamma = 1

        # self.reg = 0.001
        # self.gamma = 1

    def get_clf(self, name):
        if name == "clf_ARF":
            return ARF()
        elif name == "clf_BLS":
            return BLS(Nf=10,
                     Ne=10,
                     N1=10,
                     N2=10,
                     map_function='sigmoid',
                     enhence_function='sigmoid',
                     reg=self.reg)
        elif name == "clf_BLS_G":
            return BLS_G(Nf=10,
                     Ne=10,
                     N1=10,
                     N2=10,
                     map_function='sigmoid',
                     enhence_function='sigmoid',
                     reg=self.reg)
        elif name == "clf_BLS_pseudo":
            return BLS_pseudo(Nf=10,
                     Ne=10,
                     N1=10,
                     N2=10,
                     map_function='sigmoid',
                     enhence_function='sigmoid',
                     reg=self.reg)
        elif name == "clf_OSSBLS":
            return OSSBLS(Nf=10,
                     Ne=10,
                     N1=10,
                     N2=10,
                     map_function='sigmoid',
                     enhence_function='sigmoid',
                     reg=self.reg,
                     gamma=self.gamma,
                     n_anchor=self.n_anchor)
        elif name == "clf_OSSBLS_pseudo":
            return OSSBLS_pseudo(Nf=10,
                     Ne=10,
                     N1=10,
                     N2=10,
                     map_function='sigmoid',
                     enhence_function='sigmoid',
                     reg=self.reg,
                     gamma=self.gamma,
                     n_anchor=self.n_anchor)
        elif name == "clf_ODSSBLS":
            return ODSSBLS(
                        Nf=10,
                        Ne=10,
                        N1=10,
                        N2=10,
                        map_function='sigmoid',
                        enhence_function='sigmoid',
                        reg=self.reg,
                        gamma=self.gamma,
                        n_anchor=self.n_anchor,
                        n_class=self.n_class)
        elif name == "clf_ODSSBLS_cluster":
            return ODSSBLS_cluster(
                        Nf=10,
                        Ne=10,
                        N1=10,
                        N2=10,
                        map_function='sigmoid',
                        enhence_function='sigmoid',
                        reg=self.reg,
                        gamma=self.gamma,
                        n_anchor=self.n_anchor,
                        n_class=self.n_class)
        elif name == "clf_ISSBLS":
            return ISSBLS(
                        Nf=10,
                        Ne=10,
                        N1=10,
                        N2=10,
                        map_function='sigmoid',
                        enhence_function='sigmoid',
                        reg=self.reg,
                        gamma=self.gamma)
        elif name == "clf_ISSBLS_pseudo":
            return ISSBLS_pseudo(
                        Nf=10,
                        Ne=10,
                        N1=10,
                        N2=10,
                        map_function='sigmoid',
                        enhence_function='sigmoid',
                        reg=self.reg,
                        gamma=self.gamma)
        elif name == "clf_SOSELM":
            return SOSELM(
                        Ne=10,
                        N2=20,
                        enhence_function='sigmoid',
                        reg=self.reg,
                        gamma=self.gamma)
        elif name == 'clf_SVM':
            return svm.SVC(kernel='linear')
        elif name == 'clf_KNN':
            return KNeighborsClassifier(n_neighbors=5)
        raise ValueError("没有这个分类器")