//
// Low-energy electron background diagnostics.  Split out of FlagBkgHits
//
// Original author D. Brown
//
//
/// framework
#include "art/Framework/Principal/Event.h"
#include "fhiclcpp/ParameterSet.h"
#include "art/Framework/Principal/Handle.h"
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "art/Framework/Core/EDAnalyzer.h"
#include "Offline/GeometryService/inc/DetectorSystem.hh"
#include "art_root_io/TFileService.h"
// root
#include "TMath.h"
#include "TH1F.h"
#include "TTree.h"
#include "Math/VectorUtil.h"
// data
#include "Offline/RecoDataProducts/inc/StrawHit.hh"
#include "Offline/RecoDataProducts/inc/ComboHit.hh"
#include "Offline/RecoDataProducts/inc/StrawHitFlag.hh"
#include "Offline/DataProducts/inc/StrawIdMask.hh"
#include "Offline/RecoDataProducts/inc/BkgCluster.hh"
#include "Offline/RecoDataProducts/inc/BkgClusterHit.hh"
#include "Offline/MCDataProducts/inc/StrawDigiMC.hh"
#include "Offline/MCDataProducts/inc/PrimaryParticle.hh"
#include "Offline/MCDataProducts/inc/MCRelationship.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/DataProducts/inc/PDGCode.hh"
#include "ArtAnalysis/TrkDiag/inc/BkgHitInfo.hh"
#include "ArtAnalysis/TrkDiag/inc/BkgMCMatch.hh"
// art
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Sequence.h"
using std::string;
using namespace ROOT::Math::VectorUtil;
namespace mu2e
{
  class BkgDiag : public art::EDAnalyzer {
    public:

      struct Config {
        using Name = fhicl::Name;
        using Comment = fhicl::Comment;
        fhicl::Atom<int>              diag{                     Name("diagLevel"),                 Comment("Diag Level"),0 };
        fhicl::Atom<int>              debug{                    Name("debugLevel"),                Comment("Debug Level"),0 };
        fhicl::Atom<bool>             mcdiag{                   Name("MCDiag"),                    Comment("MonteCarlo Diag"), true };
        fhicl::Atom<bool>             hdiag{                    Name("HitDiag"),                   Comment("Hit-level Diag tuple"), false };
        fhicl::Atom<float>            maxdt{                    Name("maxTimeDifference"),         Comment("Max Time Difference"), 50.0 };
        fhicl::Atom<float>            maxdrho{                  Name("maxRhoDifference"),          Comment("Max Rho Difference"), 50.0 };
        fhicl::Atom<art::InputTag>    ComboHitCollection{       Name("ComboHitCollection"),        Comment("ComboHit collection name") };
        fhicl::Atom<art::InputTag>    BkgClusterCollection{     Name("BkgClusterCollection"),      Comment("BackgroundCluster collection name") };
        fhicl::Atom<art::InputTag>    BkgClusterHitCollection{  Name("BkgClusterHitCollection"),   Comment("BackgroundClusterHit collection name") };
        fhicl::Atom<art::InputTag>    StrawDigiMCCollection{    Name("StrawDigiMCCollection"),     Comment("StrawDigiMC collection name") };
        fhicl::Atom<art::InputTag>    MCPrimary{                Name("MCPrimary"),                 Comment("MC Primary Particle") };
        fhicl::Atom<std::string>      expectedCHLevel{          Name("ExpectedCHLevel"),           Comment("Level of the input ComboHitCollection") };
     };

      explicit BkgDiag(const art::EDAnalyzer::Table<Config>& config);
      virtual ~BkgDiag();
      virtual void beginJob();
      virtual void analyze(const art::Event& e);
    private:
      // helper functions
      void fillStrawHitInfo(size_t ich, StrawHitInfo& bkghinfo) const;
      void fillStrawHitInfoMC(StrawDigiMC const& mcdigi, art::Ptr<SimParticle>const& mptr, StrawHitInfo& shinfo) const;
      bool findData(const art::Event& e);
      void findMajority(std::vector<StrawDigiIndex>const& dids, art::Ptr<SimParticle>& mptr,XYZVectorF& mmom, int& mostCommonCode, float& fraction) const;

      // control flags
      int _diag,_debug;
      bool _mcdiag, _hdiag;
      float _maxdt, _maxdrho;
      // data tags
      art::ProductToken<ComboHitCollection> _chToken;
      art::ProductToken<BkgClusterCollection> _bkgcToken;
      art::ProductToken<BkgClusterHitCollection> _bkghToken;
      art::ProductToken<StrawDigiMCCollection> _mcdigisToken;
      art::ProductToken<PrimaryParticle> _mcprimaryToken;
      // time offset
      // cache of event objects
      const ComboHitCollection* _chcol;
      const StrawDigiMCCollection *_mcdigis;
      const PrimaryParticle *_mcprimary;
      const BkgClusterCollection *_bkgccol;
      const BkgClusterHitCollection *_bkghitcol;

      //Input Level Check
      StrawIdMask::Level expectedCHLevel_;
      std::string expectedCHLevelName_;

      // background diagnostics
      TTree* _bcdiag = 0;
      TTree* _bhdiag = 0;
      int _iev = 0;
      XYZVectorF _cpos;
      float _rmscposx = 0;
      float _rmscposy = 0;
      float _rmscrho = 0;
      float _rmscphi = 0;
      float _clusterdensity = 0;
      float _cchi2 = 0;
      float _ccons = 0;
      float _ctime = 0;
      float _cedep = 0;
      float _cpitch = 0;
      float _cyaw = 0;
      float _cqual = 0;
      float _csthqual = 0;
      float _ccomqual = 0;
      float _cecc = 0;
      float _rmsctime = 0;
      float _avecedep = 0;
      float _mindt = 0;
      float _mindrho = 0;
      bool _isinit = false;
      bool _isbkg = false;
      bool _isref = false;
      bool _isolated = false;
      bool _stereo = false;
      int _cluIdx, _nactive, _nch, _nsh, _nsth, _nsha, _nbkg;
      int _cndof;
      float _crho;
      float _zmin;
      float _zmax;
      float _zgap;
      float _zdiff;
      float _phidiff;
      float _pfrac;
      float _kQ;
      int _np, _fp, _lp, _pgap;

      // MC truth variables
      int _mpdg, _mproc;
      int _prel; // Set to -2 if a clsuter is identified as bkg after MC comparison, 0 if it is signal-like cluster, 1 if it is CE hits cluster
      int _code;
      float _frac;
      XYZVectorF _mmom, _mopos;
      int _nconv, _ndelta, _ncompt, _ngconv, _nebkg, _nprot, _nmain, _nsibling, _nrel;
      std::vector<BkgHitInfo> _bkghinfo;

      int   _nhits,_hitPdg[8192],_hitproc[8192],_hitnch[8192], _hitnsh[8192];
      std::vector<XYZVectorF> _hitPos;
      float _hitTime[8192];
  };

  BkgDiag::BkgDiag(const art::EDAnalyzer::Table<Config>& config) :
    art::EDAnalyzer{config},
    _diag( config().diag() ),
    _debug( config().debug() ),
    _mcdiag( config().mcdiag() ),
    _hdiag( config().hdiag() ),
    _maxdt( config().maxdt() ),
    _maxdrho( config().maxdrho() ),
    _chToken{ consumes<ComboHitCollection>(config().ComboHitCollection() ) },
    _bkgcToken{ consumes<BkgClusterCollection>(config().BkgClusterCollection() ) },
    _bkghToken{ consumes<BkgClusterHitCollection>(config().BkgClusterHitCollection() ) },
    _mcdigisToken{ consumes<StrawDigiMCCollection>(config().StrawDigiMCCollection() ) },
    _mcprimaryToken{ consumes<PrimaryParticle>(config().MCPrimary() ) }
  {
    StrawIdMask mask(config().expectedCHLevel());
    expectedCHLevel_ = mask.level();
    expectedCHLevelName_ = mask.levelName();
  }

  BkgDiag::~BkgDiag(){}

  void BkgDiag::beginJob() {
    art::ServiceHandle<art::TFileService> tfs;

    _iev=0;
    // detailed delta diagnostics
    _bcdiag=tfs->make<TTree>("bkgcdiag","background cluster diagnostics");
    // general branches
    _bcdiag->Branch("iev",&_iev,"iev/I");
    // cluster info branches
    _bcdiag->Branch("cpos",&_cpos);
    _bcdiag->Branch("rmscposx",&_rmscposx,"rmscposx/F");
    _bcdiag->Branch("rmscposy",&_rmscposy,"rmscposy/F");
    _bcdiag->Branch("rmscrho",&_rmscrho,"rmscrho/F");
    _bcdiag->Branch("rmscphi",&_rmscphi,"rmscphi/F");
    _bcdiag->Branch("clusterdensity",&_clusterdensity,"clusterdensity/F");
    _bcdiag->Branch("cpitch",&_cpitch,"cpitch/F");
    _bcdiag->Branch("cyaw",&_cyaw,"cyaw/F");
    _bcdiag->Branch("cqual",&_cqual,"cqual/F");
    _bcdiag->Branch("csthqual",&_csthqual,"csthqual/F");
    _bcdiag->Branch("ccomqual",&_ccomqual,"ccomqual/F");
    _bcdiag->Branch("cecc",&_cecc,"cecc/F");
    _bcdiag->Branch("cchi2",&_cchi2,"cchi2/F");
    _bcdiag->Branch("ccons",&_ccons,"ccons/F");
    _bcdiag->Branch("ctime",&_ctime,"ctime/F");
    _bcdiag->Branch("cedep",&_cedep,"cedep/F");
    _bcdiag->Branch("rmsctime",&_rmsctime,"rmsctime/F");
    _bcdiag->Branch("avecedep",&_avecedep,"avecedep/F");
    _bcdiag->Branch("isinit",&_isinit,"isinit/B");
    _bcdiag->Branch("isbkg",&_isbkg,"isbkg/B");
    _bcdiag->Branch("isref",&_isref,"isref/B");
    _bcdiag->Branch("isolated",&_isolated,"isolated/B");
    _bcdiag->Branch("stereo",&_stereo,"stereo/B");
    _bcdiag->Branch("mindt",&_mindt,"mindt/F");
    _bcdiag->Branch("mindrho",&_mindrho,"mindrho/F");
    _bcdiag->Branch("nch",&_nch,"nch/I");
    _bcdiag->Branch("nsh",&_nsh,"nsh/I");
    _bcdiag->Branch("nsth",&_nsth,"nsth/I");
    _bcdiag->Branch("nactive",&_nactive,"nactive/I");
    _bcdiag->Branch("nsha",&_nsha,"nsha/I");
    _bcdiag->Branch("nbkg",&_nbkg,"nbkg/I");
    _bcdiag->Branch("cndof",&_cndof,"cndof/I");
    _bcdiag->Branch("cluIdx",&_cluIdx,"cluIdx/I");
    // cluster hit info branch
    if(_diag > 0)
      _bcdiag->Branch("bkghinfo",&_bkghinfo);
    _bcdiag->Branch("crho",&_crho,"crho/F");
    _bcdiag->Branch("zmin",&_zmin,"zmin/F");
    _bcdiag->Branch("zmax",&_zmax,"zmax/F");
    _bcdiag->Branch("zgap",&_zgap,"zgap/F");
    _bcdiag->Branch("zdiff",&_zdiff,"zdiff/F");
    _bcdiag->Branch("phidiff",&_phidiff,"phidiff/F");
    _bcdiag->Branch("np",&_np,"np/I");
    _bcdiag->Branch("pfrac",&_pfrac,"pfrac/F");
    _bcdiag->Branch("kQ",&_kQ,"kQ/F");
    _bcdiag->Branch("fp",&_fp,"fp/I");
    _bcdiag->Branch("lp",&_lp,"lp/I");
    _bcdiag->Branch("pgap",&_pgap,"pgap/I");
    _bcdiag->Branch("nhits",&_nhits,"nhits/I");
    // mc truth branches
    if(_mcdiag){
      _bcdiag->Branch("mmom",&_mmom);
      _bcdiag->Branch("mopos",&_mopos);
      _bcdiag->Branch("mpdg",&_mpdg,"mpdg/I");
      _bcdiag->Branch("mproc",&_mproc,"mproc/I");
      _bcdiag->Branch("prel",&_prel,"prel/I");
      _bcdiag->Branch("code",&_code,"code/I");
      _bcdiag->Branch("frac",&_frac,"frac/F");
      _bcdiag->Branch("nmain",&_nmain,"nmain/I");
      _bcdiag->Branch("nsibling",&_nsibling,"nsibling/I");
      _bcdiag->Branch("nrel",&_nrel,"nrel/I");
      _bcdiag->Branch("nconv",&_nconv,"nconv/I");
      _bcdiag->Branch("ndelta",&_ndelta,"ndelta/I");
      _bcdiag->Branch("ncompt",&_ncompt,"ncompt/I");
      _bcdiag->Branch("ngconv",&_ngconv,"ngconv/I");
      _bcdiag->Branch("nebkg",&_nebkg,"nebkg/I");
      _bcdiag->Branch("nprot",&_nprot,"nprot/I");
    }
    if(_hdiag){
      _bhdiag = tfs->make<TTree>("bkghdiag","background hit diagnostics");
      _bhdiag->Branch("iev",        &_iev,          "iev/I");
      _bhdiag->Branch("nhits",      &_nhits,        "nhits/I");
      _bhdiag->Branch("pos",     &_hitPos);
      _bhdiag->Branch("time",    &_hitTime,      "hitTime[nhits]/F");
      _bhdiag->Branch("nch",  &_hitnch,    "hitnch[nhits]/I");
      _bhdiag->Branch("nsh",  &_hitnsh,    "hitnsh[nhits]/I");
      if(_mcdiag){
        _bhdiag->Branch("mcpdg",   &_hitPdg,       "hitPdg[nhits]/I");
        _bhdiag->Branch("mcproc",&_hitproc,    "hitproc[nhits]/I");
      }
    }
  }

  void BkgDiag::analyze(const art::Event& event ) {
    if(!findData(event))
      throw cet::exception("RECO")<<"mu2e::BkgDiag: data missing or incomplete"<< std::endl;
    if( !(_chcol->level() == expectedCHLevel_) ){
      throw cet::exception("RECO")<< "mu2e::BkgDiag: inconsistent outputlevel with input combo hits.\n"
                                  << "FlagBkgHits outputlevel must be "<< expectedCHLevelName_ <<" for training purposes."<< std::endl;
    }
    // loop over background clusters

    _nhits=0;
    _hitPos.clear();
    _hitPos.reserve(_chcol->size());
    for(size_t ich=0;ich<_chcol->size();++ich){
      _hitPos.push_back(_chcol->at(ich).pos());
      _hitTime[_nhits]   = _chcol->at(ich).time();
      _hitnch[_nhits] = _chcol->at(ich).nCombo();
      _hitnsh[_nhits] = _chcol->at(ich).nStrawHits();
      if(_mcdiag){
        std::vector<StrawDigiIndex> dids;
        _chcol->fillStrawDigiIndices(ich,dids);
        StrawDigiMC const& mcdigi = _mcdigis->at(dids[0]);// taking 1st digi: is there a better idea??
        auto const& sgsp = mcdigi.earlyStrawGasStep();
        if (sgsp.isNonnull()){
          art::Ptr<SimParticle> const& spp = mcdigi.earlyStrawGasStep()->simParticle();
          if(spp.isNonnull()){
            _hitPdg[_nhits] = spp->pdgId();
            _hitproc[_nhits] = spp->creationCode();
          }
          else{
            _hitPdg[_nhits] = -1;
            _hitproc[_nhits] = -1;
          }
        }
      }
      ++_nhits;
    }

    if(_hdiag)_bhdiag->Fill();

    _cluIdx=0;
    for (size_t ibkg=0;ibkg<_bkgccol->size();++ibkg){
      BkgCluster const& cluster = _bkgccol->at(ibkg);
      // fill cluster info
      _kQ = cluster.getKerasQ();
      _crho = sqrtf(cluster.pos().perp2());
      _cpos = cluster.pos();
      _ctime = cluster.time();
      _cedep = cluster.edep();
      if(cluster.getDistMethod() == BkgCluster::chi2){
        _cchi2 = cluster.points().chisquared();
        _ccons = cluster.points().consistency();
        _cndof = cluster.points().nDOF();
      }
      _isinit = cluster.flag().hasAllProperties(BkgClusterFlag::init);
      _isbkg = cluster.flag().hasAllProperties(BkgClusterFlag::bkg);
      _isref = cluster.flag().hasAllProperties(BkgClusterFlag::refined);
      _isolated = cluster.flag().hasAllProperties(BkgClusterFlag::iso);
      _stereo = cluster.flag().hasAllProperties(BkgClusterFlag::stereo);
      // info on nearest cluster
      _mindt = _mindrho = 1.0e3;
      for(size_t jbkg = 0; jbkg < _bkgccol->size(); ++jbkg){
        if(ibkg != jbkg){
          BkgCluster const& ocluster = _bkgccol->at(jbkg);
          double dt = fabs(ocluster.time() - cluster.time());
          double drho = sqrt((ocluster.pos()-cluster.pos()).Perp2());
          if(dt < _mindt) _mindt = dt;
          if(drho < _mindrho) _mindrho = drho;
        }
      }
      // fill mc info
      art::Ptr<SimParticle> mptr;
      // loop over hits in this cluster and classify them
      _nconv = 0;
      _nprot = 0;
      _nebkg = 0;
      _nmain = 0;
      _nsibling = 0;
      _mmom = XYZVectorF();
      _mopos = XYZVectorF();
      _mpdg = _mproc = 0;
      _prel=-1;
      if(_mcdiag){
        // fill vector of indices to all digis used in this cluster's hits
        // this goes recursively through the ComboHit chain
        std::vector<StrawDigiIndex> cdids;
        for(auto const& ich : cluster.hits()){
          // get the list of StrawHit indices associated with this ComboHit
          _chcol->fillStrawDigiIndices(ich,cdids);
        }
        findMajority(cdids,mptr,_mmom,_code, _frac);
        if(mptr.isNonnull()){
          _mpdg = mptr->pdgId();
          _mproc = mptr->creationCode();
          _mopos = mptr->startPosXYZ();
          _prel = BkgMCMatch::classify(mptr->creationCode());
        }
      }
      // fill cluster hit info
      _bkghinfo.clear();
      _bkghinfo.reserve(cluster.hits().size());
      _nch = cluster.hits().size();
      _nsh = _nsth = _nactive = _nsha = _nbkg = _nrel = 0;
      float sumEdep(0.), sumEcc(0.);
      float sqrSumDeltaTime(0.), sqrSumDeltaX(0.), sqrSumDeltaY(0.), sqrSumDeltaPhi(0.), sqrSumQual(0.);
      float sumPitch(0.), sumYaw(0.), sumwPitch(0.), sumwYaw(0.), sumwEcc(0.);
      float phihit(0.), phiclust(0.), phidiff(0.);
      float phimin = std::numeric_limits<float>::max();
      float phimax = -std::numeric_limits<float>::max();
      std::vector<int> strawIds;
      std::vector<int> panelIds;
      std::vector<float> hz;
      std::array<bool,StrawId::_nplanes> hp{false};
      phiclust = _cpos.phi();
      for(auto const& ich : cluster.hits()){
        ComboHit const& ch = _chcol->at(ich);
        hz.push_back(ch.pos().Z());
        hp[ch.strawId().plane()] = true;
        BkgClusterHit const& bhit = _bkghitcol->at(ich);
        sumEdep +=  ch.energyDep()/ch.nStrawHits();
        sqrSumDeltaX += std::pow(ch.pos().X() - _cpos.X(),2);
        sqrSumDeltaY += std::pow(ch.pos().Y() - _cpos.Y(),2);
        phihit = ch.pos().phi();
        phidiff = phihit- phiclust;
        if(phidiff > M_PI) phidiff -= 2*M_PI;
        if(phidiff < -M_PI) phidiff += 2*M_PI;
        if(phidiff < phimin) phimin = phidiff;
        if(phidiff > phimax) phimax = phidiff;
        sqrSumDeltaPhi += std::pow(phidiff,2);
        sqrSumDeltaTime += std::pow(ch.time() - _ctime,2);
        auto hdir = ch.hDir();
        auto wecc = ch.nStrawHits();
        sumEcc += std::sqrt(1-(ch.vVar()/ch.uVar()))*wecc;
        sumwEcc += wecc;
        if(strawIds.size() == 0 && panelIds.size() == 0){
          strawIds.push_back(ch.strawId().straw());
          panelIds.push_back(ch.strawId().uniquePanel());
        }
        else{
          int found = 0;
          for(size_t i = 0;i<strawIds.size();i++){
            if (strawIds.at(i) == ch.strawId().straw() && panelIds.at(i) == ch.strawId().uniquePanel()){
              found = 1;
            }
          }
          if(found == 0){
            strawIds.push_back(ch.strawId().straw());
            panelIds.push_back(ch.strawId().uniquePanel());
          }
        }
        if(ch.flag().hasAllProperties(StrawHitFlag::sline)){
          //quality of SLine fit
          sqrSumQual += std::pow(ch.qual(),2);

          //angle with Mu2e-Y
          float varPitch = std::pow(TMath::ACos(std::sqrt(ch.hcostVar())),2);
          float wPitch = 1/varPitch;
          float signPitch = hdir.Y()/std::abs(hdir.Y());
          sumPitch += signPitch*wPitch*hdir.theta();
          sumwPitch += wPitch;

          ROOT::Math::XYZVectorF z = {0,0,1};
          ROOT::Math::XYZVectorF dxdz = {hdir.X(),0,hdir.Z()};
          float magdxdz = std::sqrt(dxdz.Mag2());

          // angle with Mu2e-Z
          float varYaw = std::sqrt(ch.hphiVar() + varPitch);
          float wYaw = 1/varYaw;
          float signYaw = hdir.X()/std::abs(hdir.X());
          sumYaw += signYaw*wYaw*TMath::ACos(dxdz.Dot(z)/magdxdz);
          sumwYaw += wYaw;

          // # of stereo hits with SLine
          _nsth++;
        }
        _nsh += ch.nStrawHits();
        StrawHitFlag const& shf = bhit.flag();
        if(shf.hasAllProperties(StrawHitFlag::active)){
          _nactive += ch.nStrawHits();
          if(ch._mask.level() == StrawIdMask::station) _nsha+= ch.nStrawHits();
        }
        if(shf.hasAllProperties(StrawHitFlag::bkg))_nbkg+= ch.nStrawHits();
        // fill hit-specific information
        BkgHitInfo bkghinfo;
        // fill basic straw hit info
        fillStrawHitInfo(ich,bkghinfo);
        if(_mcdiag){
          std::vector<StrawDigiIndex> dids;
          _chcol->fillStrawDigiIndices(ich,dids);
          StrawDigiMC const& mcdigi = _mcdigis->at(dids[0]);
          fillStrawHitInfoMC(mcdigi,mptr,bkghinfo);
          if(bkghinfo._mrel==MCRelationship::same){
            _nmain += ch.nStrawHits(); // same as main particle
          } else if(bkghinfo._mrel>MCRelationship::sibling)
            _nsibling += ch.nStrawHits();
          if(bkghinfo._mrel>MCRelationship::none)_nrel += ch.nStrawHits();
          if(bkghinfo._mcpdg == PDGCode::proton)_nprot += ch.nStrawHits();
        }
        bkghinfo._active = shf.hasAllProperties(StrawHitFlag::active);
        bkghinfo._cbkg = shf.hasAllProperties(StrawHitFlag::bkg);
        bkghinfo._gdist = bhit.distance();
        bkghinfo._index = ich;
        // calculate separation to cluster
        auto psep = ch.pos()-cluster.pos();
        auto pdir = PerpVector(psep,GenVector::ZDir()).Unit();
        bkghinfo._rpos = psep;
        bkghinfo._rerr = std::max(float(2.5),ch.posRes(ComboHit::wire)*fabs(pdir.Dot(ch.uDir())));
        _bkghinfo.push_back(bkghinfo);
      }
      _avecedep = sumEdep/_nch;
      _cecc = sumEcc/sumwEcc;
      _rmscposx = std::sqrt(sqrSumDeltaX/_nch);
      _rmscposy = std::sqrt(sqrSumDeltaY/_nch);
      _rmscrho = std::sqrt((sqrSumDeltaX+sqrSumDeltaY)/_nch);
      _rmscphi = std::sqrt(sqrSumDeltaPhi/_nch);
      _rmsctime = std::sqrt(sqrSumDeltaTime/_nch);
      float areaXY = M_PI*_rmscposx*_rmscposy;
      if(areaXY > 0)
        _clusterdensity = static_cast<float>(_nch)/areaXY;
      else
        _clusterdensity = 0.0;
      _cpitch = _nsth > 0 ? sumPitch/sumwPitch : 0.;
      _cyaw = _nsth > 0 ? sumYaw/sumwYaw : 0.;
      _cqual = std::sqrt(sqrSumQual/_nch); // average SLine fit quality of the cluster
      _csthqual = float(_nsth)/float(_nch); // SLine ratio the cluster
      _ccomqual = _cqual + _csthqual; // combined quality metric
      std::sort(hz.begin(),hz.end());
      _zgap = 0.0;
      _zdiff = 0.0;
      _phidiff = 0.0;
      _zmin = 0.0;
      _zmax = 0.0;
      for (unsigned iz=1;iz<hz.size();++iz) _zgap = std::max(_zgap,hz[iz]-hz[iz-1]);
      if(!hz.empty()){
        _zmin = hz.front();
        _zmax = hz.back();
      }
      if(_nch >= 2) {
        _zdiff = _zmax - _zmin;
        _phidiff = phimax - phimin;
      }
      _lp = -1; // last plane in cluster
      _fp = StrawId::_nplanes; // first plane in cluster
      _np = 0; // # of planes
      _pgap = 0; // largest plane gap
      int lp(-1); // last plane seen
      for(int ip=0;ip < StrawId::_nplanes; ++ip){
        if(hp[ip]){
          _np++;
          if(lp > 0 && ip - lp -1 > _pgap)_pgap = ip - lp -1;
          if(ip > _lp)_lp = ip;
          if(ip < _fp)_fp = ip;
          lp = ip;
        }
      }
      _pfrac = (_lp != _fp) ? static_cast<float>(_np)/static_cast<float>(_lp - _fp) : 1.0f; 
      _bcdiag->Fill();
      ++_cluIdx;
    }
    ++_iev;
  }

  bool BkgDiag::findData(const art::Event& evt){
    _chcol = 0; _bkgccol = 0; _mcdigis = 0;
    auto chH = evt.getValidHandle(_chToken);
    _chcol = chH.product();
    auto bkgcH = evt.getValidHandle(_bkgcToken);
    _bkgccol = bkgcH.product();
    auto bkghH = evt.getValidHandle(_bkghToken);
    _bkghitcol = bkghH.product();
    if(_mcdiag){
      auto mcdH = evt.getValidHandle(_mcdigisToken);
      _mcdigis = mcdH.product();
      auto mcpH = evt.getValidHandle(_mcprimaryToken);
      _mcprimary = mcpH.product();
    }
    return _chcol != 0 && _bkgccol != 0
      && ( (_mcdigis != 0 && _mcprimary != 0)  || !_mcdiag);
  }

  // This function performs a two-stage truth-matching validation:
  // 1. Process Majority (>50%): The cluster is assigned a 'mostCommonCode' if more than half of the StrawDigis in the cluster share that same creation code.
  // 2. Particle Purity (>80%): To avoid assigning a representative SimParticle (mptr) * and momentum (mmom) to a "fragmented" cluster, a single particle
  // must account for > 80% of the hits belonging to that majority process.
  // If these conditions are not met, mptr is returned as a null Ptr and mmom is set to zero.
  void BkgDiag::findMajority(std::vector<StrawDigiIndex> const& dids, art::Ptr<SimParticle>& mptr, XYZVectorF& mmom, int& mostCommonCode, float& fraction) const {
    // Reset outputs
    mptr = art::Ptr<SimParticle>();
    mmom = XYZVectorF();
    mostCommonCode = -1;
    fraction = 0.0;

    if (dids.empty()) return;

    // 1. Data structures to track process and individual particles
    std::map<int, unsigned> codeCounts;
    std::map<art::Ptr<SimParticle>, unsigned> particleCounts;
    std::map<art::Ptr<SimParticle>, XYZVectorF> particleMoms;

    for (auto id : dids) {
      StrawDigiMC const& mcdigi = _mcdigis->at(id);
      auto const& sgsp = mcdigi.earlyStrawGasStep();
      if (sgsp.isNonnull()) {
        art::Ptr<SimParticle> const& spp = sgsp->simParticle();
        if (spp.isNonnull()) {
          int c = spp->creationCode();
          codeCounts[c]++;
          particleCounts[spp]++;
          // Store momentum of the first time we see this specific particle
          if (particleMoms.find(spp) == particleMoms.end()) {
            particleMoms[spp] = sgsp->momentum();
          }
        }
      }
    }

    // 2. Find the winning creation code
    int bestCode = -1;
    unsigned maxCodeCount = 0;
    for (auto const& [code, count] : codeCounts) {
      if (count > maxCodeCount) {
        maxCodeCount = count;
        bestCode = code;
      }
    }

    // 3. Evaluate Condition 1: Creation Code Majority > 50%
    float codeFrac = static_cast<float>(maxCodeCount) / static_cast<float>(dids.size()); 
    if (bestCode != -1 && codeFrac > 0.5) {
      // 4. Find the dominant particle WITHIN that winning code
      art::Ptr<SimParticle> bestPart;
      unsigned maxPartCount = 0;

      for (auto const& [spp, count] : particleCounts) {
        if (spp->creationCode() == bestCode) {
          if (count > maxPartCount) {
            maxPartCount = count;
            bestPart = spp;
          }
        }
      }

      // 5. Evaluate Condition 2: Particle represents > 80% of that code's hits
      float partInCodeFrac = static_cast<float>(maxPartCount) / static_cast<float>(maxCodeCount);
      if (bestPart.isNonnull() && partInCodeFrac > 0.8) {
        mptr = bestPart;
        mmom = particleMoms[bestPart];
        mostCommonCode = bestCode;
        fraction = codeFrac; // Reporting the process fraction
      }
    }
  }

  void BkgDiag::fillStrawHitInfoMC(StrawDigiMC const& mcdigi, art::Ptr<SimParticle>const& mptr, StrawHitInfo& shinfo) const {
    // use early step to define the MC match
    auto const& sgsp = mcdigi.earlyStrawGasStep();
    art::Ptr<SimParticle> const& spp = sgsp->simParticle();
    shinfo._mctime = sgsp->time();
    shinfo._mcht = mcdigi.wireEndTime(mcdigi.earlyEnd());
    shinfo._mcpdg = spp->pdgId();
    shinfo._mcproc = spp->creationCode();
    shinfo._mcedep = mcdigi.energySum();
    shinfo._mcpos = sgsp->position();
    shinfo._mcmom = sqrt(sgsp->momentum().mag2());
    double cosd = cos(sgsp->momentum().Theta());
    shinfo._mctd = cosd/sqrt(1.0-cosd*cosd);
    // relationship to main particle
    shinfo._mrel=MCRelationship::none;
    if(sgsp.isNonnull() && mptr.isNonnull()){
      art::Ptr<SimParticle> const& spp = sgsp->simParticle();
      if(spp.isNonnull()){
        MCRelationship rel(spp,mptr);
        shinfo._prel = BkgMCMatch::classify(spp->creationCode());
        shinfo._mrel = rel.relationship();
      }
    }
  }

  void BkgDiag::fillStrawHitInfo(size_t ich, StrawHitInfo& shinfo) const {
    ComboHit const& ch = _chcol->at(ich);
    auto const& shf = ch.flag();

    shinfo._stereo = shf.hasAllProperties(StrawHitFlag::stereo);
    shinfo._tdiv = shf.hasAllProperties(StrawHitFlag::tdiv);
    shinfo._esel = shf.hasAllProperties(StrawHitFlag::energysel);
    shinfo._rsel = shf.hasAllProperties(StrawHitFlag::radsel);
    shinfo._tsel = shf.hasAllProperties(StrawHitFlag::timesel);
    shinfo._strawxtalk = shf.hasAllProperties(StrawHitFlag::strawxtalk);
    shinfo._elecxtalk = shf.hasAllProperties(StrawHitFlag::elecxtalk);
    shinfo._isolated = shf.hasAllProperties(StrawHitFlag::isolated);
    shinfo._bkg = shf.hasAllProperties(StrawHitFlag::bkg);
    shinfo._bkgc = shf.hasAllProperties(StrawHitFlag::bkgclust);

    shinfo._pos = ch.pos();
    shinfo._time = ch.correctedTime();
    shinfo._wdist = ch.wireDist();
    shinfo._wres = ch.posRes(ComboHit::wire);
    shinfo._tres = ch.posRes(ComboHit::trans);
    // info depending on stereo hits
    shinfo._chisq = ch.qual();
    shinfo._edep = ch.energyDep();
    StrawId const& sid = ch.strawId();
    shinfo._plane = sid.plane();
    shinfo._panel = sid.panel();
    shinfo._layer = sid.layer();
    shinfo._straw = sid.straw();
    shinfo._stereo = ch.flag().hasAllProperties(StrawHitFlag::stereo);
    shinfo._tdiv = ch.flag().hasAllProperties(StrawHitFlag::tdiv);
  }
} // mu2e namespace

// Part of the magic that makes this class a module.
using mu2e::BkgDiag;
DEFINE_ART_MODULE(BkgDiag)
