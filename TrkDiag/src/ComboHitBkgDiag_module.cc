#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "fhiclcpp/ParameterSet.h"
#include "art_root_io/TFileService.h"

#include "Offline/RecoDataProducts/inc/ComboHit.hh"
#include "Offline/RecoDataProducts/inc/StrawHitFlag.hh"
#include "Offline/MCDataProducts/inc/StrawDigiMC.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "ArtAnalysis/TrkDiag/inc/BkgMCMatch.hh"

#include "TTree.h"

namespace mu2e{

  class ComboHitBkgDiag : public art::EDAnalyzer {
  public:
    struct Config {
      fhicl::Atom<art::InputTag> ComboHitCollection { fhicl::Name("ComboHitCollection") };
      fhicl::Atom<art::InputTag> StrawDigiMCCollection { fhicl::Name("StrawDigiMCCollection") };
    };

    explicit ComboHitBkgDiag(const art::EDAnalyzer::Table<Config>& config);

    void beginJob() override;
    void analyze(const art::Event& e) override;
    void endJob() override;

  private:
    art::ProductToken<ComboHitCollection> _chToken;
    art::ProductToken<StrawDigiMCCollection> _mcdigiToken;

    const ComboHitCollection* _chcol = nullptr;
    const StrawDigiMCCollection* _mcdigis = nullptr;

    TTree* _tree = nullptr;

    int _iev;
    int _creationCode;
    int _pdg;
    bool _flaggedBkg;
    int _totalce = 0;
    int _totalbkg = 0;
    int _totalsig = 0;
    int _cemis = 0;
    int _bkgtrue = 0;
    int _sigmis = 0;
  };

  ComboHitBkgDiag::ComboHitBkgDiag(const art::EDAnalyzer::Table<Config>& config) :
    art::EDAnalyzer{config},
    _chToken(consumes<ComboHitCollection>(config().ComboHitCollection())),
    _mcdigiToken(consumes<StrawDigiMCCollection>(config().StrawDigiMCCollection()))
  {}

  void ComboHitBkgDiag::beginJob() {
    art::ServiceHandle<art::TFileService> tfs;
    _tree = tfs->make<TTree>("chbkg", "ComboHit background truth check");
    _tree->Branch("iev", &_iev, "iev/I");
    _tree->Branch("creationCode", &_creationCode, "creationCode/I");
    _tree->Branch("pdg", &_pdg, "pdg/I");
    _tree->Branch("flaggedBkg", &_flaggedBkg, "flaggedBkg/O");
  }

  void ComboHitBkgDiag::analyze(const art::Event& event) {
    _iev = event.id().event();
    _chcol = event.getValidHandle(_chToken).product();
    _mcdigis = event.getValidHandle(_mcdigiToken).product();

    for (size_t ich = 0; ich < _chcol->size(); ++ich) {
      ComboHit const& ch = _chcol->at(ich);

      _pdg = 0;
      _creationCode = -1;
      _flaggedBkg = ch.flag().hasAllProperties(StrawHitFlag::bkg);
      std::vector<StrawDigiIndex> dids;
      _chcol->fillStrawDigiIndices(ich, dids);

      if(dids.empty()){
        _tree->Fill();
        continue;
      }
      // Only looking at the first StrawDigi to find the MC true particle info
      // Using first StrawDigi as a representative of the ComboHit
      // This logic could be refined but it works for now
      StrawDigiMC const& mcdigi = _mcdigis->at(dids[0]);
      auto const& sgsp = mcdigi.earlyStrawGasStep();
      if(sgsp.isNonnull()){
        art::Ptr<SimParticle> const& sp = sgsp->simParticle();
        if(sp.isNonnull()){ 
          _pdg = sp->pdgId();
          ProcessCode pCode = sp->creationCode();
          _creationCode = pCode;

          if(BkgMCMatch::isCE(pCode))
            _totalce += ch.nStrawHits();
          else if(BkgMCMatch::isBackground(pCode))
            _totalbkg += ch.nStrawHits();
          else
            _totalsig += ch.nStrawHits();

          if(_flaggedBkg){
            if(BkgMCMatch::isBackground(pCode))
              _bkgtrue += ch.nStrawHits();
            else if(BkgMCMatch::isCE(pCode))
              _cemis += ch.nStrawHits();
            else
              _sigmis += ch.nStrawHits();
          }
        }
      }
      //The tree is filled per ComboHit
      _tree->Fill();
    }
  }

  void ComboHitBkgDiag::endJob() {
    std::cout << "====================================\n";
    std::cout << " FINAL COUNTS OVER ALL EVENTS\n";
    std::cout << " Total background hits = " << _totalbkg <<" True background hits = " << _bkgtrue << "\n";
    std::cout << " Total CE hits = " << _totalce <<" CE mis-tagged hits = " << _cemis   << "\n";
    std::cout << " Total Signal hits = "<< _totalsig <<" Signal mis-tagged hits  = " << _sigmis  << "\n";
    std::cout << "====================================\n";
  }
}
using mu2e::ComboHitBkgDiag;
DEFINE_ART_MODULE(ComboHitBkgDiag)
