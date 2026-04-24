cet_build_plugin(CrvInference art::module
    REG_SOURCE src/CrvInference_module.cc
    LIBRARIES REG
      Offline::RecoDataProducts
      Offline::DataProducts
      Offline::ConfigTools
      xgboost
)

cet_build_plugin(CrvInferenceAna art::module
    REG_SOURCE src/CrvInferenceAna_module.cc
    LIBRARIES REG
      Offline::RecoDataProducts
      art_root_io::TFileService_service
      ROOT::Hist
)

install_fhicl(SUBDIRS fcl SUBDIRNAME ArtAnalysis/CrvInference/fcl)
install(DIRECTORY data DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/ArtAnalysis/CrvInference)
install_source(SUBDIRS src)
