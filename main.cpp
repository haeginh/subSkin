#include <igl/arap.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/readOFF.h>
#include <igl/readPLY.h>
#include <igl/writePLY.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/doublearea.h>
#include <igl/barycenter.h>
#include <igl/readTGF.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/bbw.h>
#include <igl/boundary_conditions.h>

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd V_uv;
Eigen::MatrixXd initial_guess;


int main(int argc, char *argv[])
{
  using namespace std;
  // Load a mesh in OFF format
  igl::readPLY("../skin.ply", V, F);
  Eigen::VectorXd A;
  igl::doublearea(V, F, A);
  Eigen::VectorXd W = Eigen::VectorXd::Zero(V.rows());
  for(int i=0;i<F.rows();i++)
  {
    W(F(i, 0)) += A(i);
    W(F(i, 1)) += A(i);
    W(F(i, 2)) += A(i);
  }
  W = W/W.sum();

  std::map<double, int> cdf;
  double sum(0);
  for(int i=0;i<W.rows();i++)
  {
    sum += W(i);
    cdf[sum] = i;
  }

  double unitA = 25.;
  int num = floor(A.sum()*0.5/unitA);
  std::cout<<num<<std::endl;
  set<int> selected;
  Eigen::VectorXd rand = Eigen::VectorXd::Random(num*2).cwiseAbs();
  for(int i=0;selected.size()<num;i++)
    selected.insert(cdf.upper_bound(rand(i))->second);

  vector<int> selectedVec(selected.begin(), selected.end());
  Eigen::MatrixXd Vsel(num, 3);
  for(int i;i<num;i++) Vsel.row(i) = V.row(selectedVec[i]);
  Eigen::MatrixXi Ftmp;
  igl::writePLY("selected.ply", Vsel, Ftmp);

  //bbw
  Eigen::MatrixXd C, VT;
  Eigen::MatrixXd V1 = V;
  Eigen::MatrixXi BE;
  igl::readTGF("../MRCP_AM.tgf", C, BE);
  V1.conservativeResize(V1.rows() + C.rows(), 3);
  V1.bottomRows(C.rows()) = C;
  for (int i = 0; i < BE.rows(); i++)
  {
    int num1 = (C.row(BE(i, 0)) - C.row(BE(i, 1))).norm();
    Eigen::RowVector3d itvl = (C.row(BE(i, 1)) - C.row(BE(i, 0))) / (double)num1;
    Eigen::MatrixXd boneP(num1 - 1, 3);
    for (int n = 1; n < num1; n++)
      boneP.row(n - 1) = C.row(BE(i, 0)) + n * itvl;
    V1.conservativeResize(V1.rows() + num1 - 1, 3);
    V1.bottomRows(num1 - 1) = boneP;
  }
  Eigen::MatrixXi FT, TT;
  igl::copyleft::tetgen::tetrahedralize(V1, F, "qp/0.0001YT0.000000001", VT, TT, FT);
  Eigen::MatrixXd bc, W1;
  Eigen::VectorXi b;
  igl::boundary_conditions(VT, TT, C, Eigen::VectorXi(), BE, Eigen::MatrixXi(), b, bc);
  igl::BBWData bbw_data;
  bbw_data.active_set_params.max_iter = 10;
  bbw_data.verbosity = 2;
  Eigen::VectorXi cvtBone(BE.rows());
  cvtBone<<0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 3, 3, 3, 0, 4, 4, 4;
  Eigen::MatrixXd bc1 = Eigen::MatrixXd::Zero(bc.rows(), 5);
  for(int i=0;i<cvtBone.rows();i++)
    bc1.col(cvtBone(i)) += bc.col(i);
  igl::normalize_row_sums(bc1, bc1);
  igl::bbw(VT, TT, b, bc1, bbw_data, W1);
  Eigen::VectorXi boneID(selectedVec.size());
  for(int i=0;i<selectedVec.size();i++)
  {
    Eigen::Index idx;
    W1.row(selectedVec[i]).transpose().maxCoeff(&idx);
    boneID(i) = idx;
  }
  // W1.rowwise().maxCoeff();

  Eigen::MatrixXd BC;
  igl::barycenter(V, F, BC);
  vector<vector<int>> group(num);
  Eigen::MatrixXd C1 = Eigen::MatrixXd::Random(num, 3);
  Eigen::VectorXi ID(F.rows());
  ///simple distance///
  for(int i=0;i<BC.rows();i++)
  {
    Eigen::Index idxF, idx;
    int idxV(-1);
    (W1.row(F(i,0))+W1.row(F(i,1))+W1.row(F(i,2))).transpose().maxCoeff(&idxF);
    Eigen::VectorXd dist2 = (Vsel.rowwise() - BC.row(i)).cwiseAbs2().rowwise().sum();
    
    while(idxV!=idxF)
    {
      if(idxV>=0)
      {
        dist2(idx) = __DBL_MAX__;
      }
      dist2.minCoeff(&idx);
      idxV = boneID(idx);
    } 
    group[(int)idx].push_back(i);
    ID(i) = idx;
  }
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.data().set_colors(igl::slice(C1, ID, 1));
  viewer.launch();
  
  ofstream ofs("div_skin.obj");
  for(int i=0;i<V.rows();i++)
    ofs<<"v "<<V.row(i)<<endl;
  for(int g=0;g<group.size();g++)
  {
    if(group[g].empty()) continue;
    ofs<<endl<<"g "<<g<<endl<<"usemtl "<<g<<endl<<"s"<<endl<<endl;
    for(int f:group[g])
      ofs<<"f "<<F(f,0)+1<<" "<<F(f,1)+1<<" "<<F(f,2)+1<<endl;
  }
  ofs.close();

  //read ele
  ifstream ifs_ele("../MRCP_AM.ele");
  int numE, dump;
  ifs_ele>>numE>>dump>>dump;
  TT.resize(numE, 5);
  int r(0);
  for(int i=0;i<numE;i++)
  {
    int a, b, c, d, id;
    ifs_ele>>dump>>a>>b>>c>>d>>id;
    if(id<12200 || id>12501)
      TT.row(r++)<<a, b, c, d, id;
  }
  numE = r;
  TT.conservativeResize(numE, 5);
  Eigen::MatrixXi TT1 = TT.leftCols(4);
  std::vector<int> vNums(TT1.data(),TT1.data()+4*TT1.rows());
  sort(vNums.begin(), vNums.end());
  vNums.erase(unique(vNums.begin(), vNums.end()), vNums.end());
  cout<<"read ELE file"<<endl;

  //read node
  ifstream ifs_node("../MRCP_AM.node");
  int numN;
  ifs_node>>numN>>dump>>dump>>dump;
  VT.resize(vNums.size(), 3);
  map<int, int> whole2ext;
  for(int i=0, n=0;i<numN;i++)
  {
    double x, y, z;
    ifs_node>>dump>>x>>y>>z;
    if(i==vNums[n])
    {
      VT.row(n) << x, y, z;
      whole2ext[i] = n;
      n++;
    }
  }
  numN = vNums.size();
  cout<<"read NODE file"<<endl;
  for(int i=0;i<TT.rows();i++)
  {
    TT(i,0) = whole2ext[TT(i,0)];
    TT(i,1) = whole2ext[TT(i,1)];
    TT(i,2) = whole2ext[TT(i,2)];
    TT(i,3) = whole2ext[TT(i,3)];
  }
  cout<<"converted ELE"<<endl;

  Eigen::Index idx;
  (VT.rowwise()-V.row(0)).rowwise().squaredNorm().minCoeff(&idx);
  if((VT.middleRows(idx, V.rows()) - V).cwiseAbs2().sum()>0.01) 
  {
    cout<<"RST was not matched (sum[diff2]: "<<(VT.middleRows(idx, V.rows()) - V).cwiseAbs2().sum()<<")"<<endl;
    return 1;
  }

  //generate skin layers
  VT.conservativeResize(numN+3*V.rows(), 3);
  Eigen::MatrixXd N;
  igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, N);
  V += 0.1605633*N;
  VT.middleRows(numN, V.rows()) = V; //100
  igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, N);
  VT.middleRows(numN+V.rows(), V.rows()) = V + 0.005*N; //50
  VT.middleRows(numN+2*V.rows(), V.rows()) = V + 0.01*N;  //outer
  cout<<"generated layer nodes"<<endl;
  //generate elements
  //-(id*10): non-target / -(id*10+1): target
  TT.conservativeResize(numE + F.rows()*3*3, 5);
  Eigen::MatrixXi bottomT = Eigen::MatrixXi::Zero(F.rows()*3, 4);
  Eigen::MatrixXi topT = Eigen::MatrixXi::Zero(F.rows()*3, 4);
  Eigen::VectorXi idCol(F.rows()*3);
  Eigen::MatrixXi bottomBool = Eigen::MatrixXi::Zero(F.rows()*3, 4);
  for(int i=0;i<F.rows();i++)
  {
    idCol.middleRows(i*3, 3) = Eigen::Vector3i::Constant(ID(i));
    bottomBool.row(i*3) = Eigen::RowVector4i(0, 1, 1, 1);
    bottomBool.row(i*3+1) = Eigen::RowVector4i(0, 0, 1, 1);
    bottomBool.row(i*3+2) = Eigen::RowVector4i(0, 0, 0, 1);
    bottomT.row(i*3) = Eigen::RowVector4i(0, F(i, 0), F(i, 1), F(i, 2));
    bottomT.row(i*3+1) = Eigen::RowVector4i(0, 0, F(i, 0), F(i, 2));
    bottomT.row(i*3+2) = Eigen::RowVector4i(0, 0, 0, F(i, 0));
    topT.row(i*3) = Eigen::RowVector4i(F(i, 1), 0, 0, 0);
    topT.row(i*3+1) = Eigen::RowVector4i(F(i, 1), F(i, 2), 0, 0);
    topT.row(i*3+2) = Eigen::RowVector4i(F(i, 0), F(i, 1), F(i, 2), 0);
  }
  Eigen::MatrixXi topBool = (bottomBool.array()-1).abs();

  //rst-100
  TT.middleRows(numE, F.rows()*3).leftCols(4) = (bottomT.array()+idx)*bottomBool.array();
  TT.middleRows(numE, F.rows()*3).leftCols(4) += ((topT.array()+numN)*topBool.array()).matrix();
  TT.middleRows(numE, F.rows()*3).rightCols(1) = -idCol*10;
  //100-50
  TT.middleRows(numE+F.rows()*3, F.rows()*3).leftCols(4) = (bottomT.array()+numN)*bottomBool.array();
  TT.middleRows(numE+F.rows()*3, F.rows()*3).leftCols(4) += ((topT.array()+numN+V.rows())*topBool.array()).matrix();
  TT.middleRows(numE+F.rows()*3, F.rows()*3).rightCols(1) = -idCol.array()*10-1;
  //50-outer
  TT.middleRows(numE+F.rows()*6, F.rows()*3).leftCols(4) = (bottomT.array()+numN+V.rows())*bottomBool.array();
  TT.middleRows(numE+F.rows()*6, F.rows()*3).leftCols(4) += ((topT.array()+numN+V.rows()*2)*topBool.array()).matrix();
  TT.middleRows(numE+F.rows()*6, F.rows()*3).rightCols(1) = -idCol*10;
  cout<<"generated layer elements"<<endl;

  ofstream ofs_ELE("skin_div.ele");
  Eigen::MatrixXi TT_p(TT.rows(), 6);
  TT_p<< Eigen::VectorXi::LinSpaced(TT.rows(), 0, TT.rows()-1), TT;
  ofs_ELE<<TT.rows()<<"   4   1"<<endl;
  ofs_ELE<<TT_p<<endl;
  ofs_ELE.close();
  ofstream ofs_NODE("skin_div.node");
  ofs_NODE<<VT.rows()<<"   3   0   0"<<endl;
  numN = log10(VT.rows())+1;
  for(int i=0;i<VT.rows();i++)
    ofs_NODE<<fixed<<setprecision(20)<<setw(numN)<<i<<" "<<VT.row(i)<<endl;
  ofs_NODE.close();
  return 0;


}
