#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <windows.h>
#include <string>
#include <filesystem>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/memory.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/geometry/polygon_mesh.h>
#include <pcl/geometry/mesh_conversion.h>
#include <pcl/geometry/triangle_mesh.h>
#include <pcl/registration/icp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/normal_3d_omp.h>  // OpenMP pour NormalEstimation
#include <pcl/features/fpfh_omp.h>       // OpenMP pour FPFH
#include <future>  // Pour exécuter les tâches en parallèle
#include <pcl/registration/icp_nl.h>
#include <omp.h>
//----------------------------------------------------------------------------------------------
#include <cstdint>
#include <fstream>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/IO/OBJ.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Surface_mesh/IO/PLY.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Polygon_mesh_processing/locate.h>
#include <cmath>
#include <CGAL/Point_set_3.h>
#include <CGAL/jet_estimate_normals.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/shape_detection.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/distance.h>
#include <ctime>
using namespace std;
using namespace pcl;
int user_data;
typedef CGAL::Simple_cartesian<double>  Kernel;
typedef Kernel::Point_3    Point;
typedef Kernel::Vector_3 Vector3;
typedef CGAL::Surface_mesh<Point> mesh;
typedef boost::graph_traits<mesh>::face_descriptor face_descriptor;
typedef boost::graph_traits<mesh>::vertex_descriptor vertex_descriptor;
typedef mesh::Vertex_index Vertex_index;
typedef mesh::Face_index Face_index;
typedef mesh::Halfedge_index Halfedge_index;
typedef Kernel::Iso_cuboid_3 Cuboid;
typedef Kernel::FT FT;  // Type pour les distances
// Définir le type pour l'arbre AABB
typedef CGAL::AABB_face_graph_triangle_primitive<mesh> Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;
//def pour le point set et le calcul de courbure
typedef CGAL::Point_set_3<Point> Point_set;
Point_set points;
//pour pcl 
pcl::PolygonMesh cloud2;
PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
PointCloud<PointXYZ>::Ptr cloudsec(new PointCloud<PointXYZ>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudXYZRG(new pcl::PointCloud<pcl::PointXYZRGB>());

map<face_descriptor, double> mapdens ;// map de la densité de point en fonction de la face
map<Point, double> mappointdens; // map de la densité des point en fonction d'un point du maillage
map<face_descriptor, double> mapdistance ;//map de la distance moyenne des point a une face
map<Point, double> mapdistpoint;//map distance moyenne a un point maillage
map<face_descriptor, list<Point>> mapFaces;
Eigen::Matrix4f transfo;
//---------
mesh test;


pcl::PolygonMesh convert_cgal_to_pcl(const mesh cgal_mesh) {
    pcl::PolygonMesh pcl_mesh;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    // Convertir les sommets
    for (auto v : cgal_mesh.vertices()) {
        Point p = cgal_mesh.point(v);
        pcl_cloud->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
    }

    pcl::toPCLPointCloud2(*pcl_cloud, pcl_mesh.cloud);

    // Convertir les faces
    for (auto f : cgal_mesh.faces()) {
        pcl::Vertices vertices;
        for (auto v : CGAL::vertices_around_face(cgal_mesh.halfedge(f), cgal_mesh)) {
            vertices.vertices.push_back(v.idx());
        }
        pcl_mesh.polygons.push_back(vertices);
    }

    return pcl_mesh;
}


void setPointColor(pcl::PointXYZRGB& point, int r, int g, int b) {
    uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
        static_cast<uint32_t>(g) << 8 |
        static_cast<uint32_t>(b));
    point.rgb = *reinterpret_cast<float*>(&rgb);
}


void getColorFromValue(float value, float min_val, float max_val, int* r, int* g, int* b) {
    if (value < 0) {
        // Valeurs négatives : Bleu → Blanc
        double ratio = (value - min_val) / (-min_val);
        ratio = std::clamp(ratio, 0.0, 1.0);

        *r = static_cast<int>(255.0 * ratio);
        *g = static_cast<int>(255.0 * ratio);
        *b = 255;
    }
    else {
        // Valeurs positives : Blanc → Rouge
        double ratio = value / max_val;
        ratio = std::clamp(ratio, 0.0, 1.0);

        *r = 255;
        *g = static_cast<int>(255.0 * (1.0 - ratio));
        *b = static_cast<int>(255.0 * (1.0 - ratio));
    }
}



void convertAndColorMesh(pcl::PolygonMesh& mesh, std::map<Point, double> dico, float min_val, float max_val) {
    // Convertir sensor_msgs::PointCloud2 en pcl::PointCloud<pcl::PointXYZ>
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(cloud2.cloud, *cloudXYZ);

    // Créer un nuage coloré
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudXYZRGB(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloudXYZRGB->points.resize(cloudXYZ->points.size());
    cloudXYZRGB->width = cloudXYZ->width;
    cloudXYZRGB->height = cloudXYZ->height;
    cloudXYZRGB->is_dense = cloudXYZ->is_dense;

    // Copier les points et ajouter la couleur
    for (size_t i = 0; i < cloudXYZ->points.size(); ++i) {
        cloudXYZRGB->points[i].x = cloudXYZ->points[i].x;
        cloudXYZRGB->points[i].y = cloudXYZ->points[i].y;
        cloudXYZRGB->points[i].z = cloudXYZ->points[i].z;
        int r=255;
        int g=255;
        int b=255;
        Point lePoint(cloudXYZ->points[i].x, cloudXYZ->points[i].y, cloudXYZ->points[i].z);
        

        getColorFromValue(dico[lePoint], min_val, max_val, &r, &g, &b);
        setPointColor(cloudXYZRGB->points[i], r, g, b);
       // cout << "la point est" << lePoint << "la valeur est de  : " << dico[lePoint] <<"valeur des courleurs "<<r<<": "<<g <<": "<<b <<" :" << endl;

    }


    pcl::toPCLPointCloud2(*cloudXYZRGB, cloud2.cloud);
}



pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertToPointXYZRGB(pcl::PointCloud<pcl::PointXYZ>::Ptr cl, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Parcourir chaque point et copier les coordonnées avec la couleur spécifiée
    for (const auto& point : cl->points) {
        pcl::PointXYZRGB point_rgb;
        point_rgb.x = point.x;
        point_rgb.y = point.y;
        point_rgb.z = point.z;
        point_rgb.r = r;
        point_rgb.g = g;
        point_rgb.b = b;
        cloud_rgb->points.push_back(point_rgb);
    }

    cloud_rgb->width = cl->width;
    cloud_rgb->height = cl->height;
    cloud_rgb->is_dense = cl->is_dense;

    return cloud_rgb;
}


void
loadingPlyAndPcdView(mesh poly)
{
    //création des empty
    cloud2 = convert_cgal_to_pcl(poly);
    //création du visualizer
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    //chargement du point cloud
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud(cloudXYZRG,"cl",0);
     viewer->addPolygonMesh(cloud2, "mesh", 0);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5.0, "cl");
    //intialisation du viewer
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPosition(0,0,0, 0, 0, 0, 0);
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}



/*
   
    donne la liste d'appartennance des points a chaque face le faire dans test face faire une map comme clé id face et comme second liste de points
    verifier qu'un point n'apparait pas plusieurs 

*/

double testFace(mesh poly, PointCloud<PointXYZ>::Ptr nuage)
{
    double score=0;
    double nombrepoint = 0;
    double dist;
    // Construire l'arbre AABB
    Tree tree(faces(poly).first, faces(poly).second, poly);
    tree.accelerate_distance_queries();
    // Parcourir tous les points du nuage
    for (const auto& point : nuage->points) 
    {
        nombrepoint++;
        // Point à localiser
        Point query(point.x,point.y,point.z);
       
        Tree::Point_and_primitive_id closest = tree.closest_point_and_primitive(query);
        Point closest_point = closest.first;
        face_descriptor closest_face = closest.second;
        mapFaces[closest_face].push_back(query);
        if (closest_face != boost::graph_traits<mesh>::null_face()) {

            

            mapdens[closest_face]++;
            mappointdens[closest_point]++;
            // Calculer la distance au carré entre le point et le point le plus proche sur la face
            FT sq_distance = CGAL::squared_distance(query, closest_point);
            //on calcul les deux vecteur pour faire le produit scalaire et ainsi obtenir le signe pour la distance
            Vector3 vecteurDist = closest_point - query;
            Vector3 normalFace = CGAL::Polygon_mesh_processing::compute_face_normal(closest_face, poly);
            FT  signe = CGAL::scalar_product(vecteurDist, normalFace);
            if (signe < 0)
            {
                signe = -1.0;
            }
            else
            {
                signe = 1.0;
            }
            dist = std::sqrt(sq_distance) * signe;
            mapdistance[closest_face] += dist;

            for (auto v : CGAL::vertices_around_face(poly.halfedge(closest_face), poly)) {
                FT distancept = CGAL::squared_distance(query, poly.point(v));
                //---------------------------------------------------------
                if (mapdistpoint.find(poly.point(v)) != mapdistpoint.end()) {
                
                    if (mapdistpoint[poly.point(v)] > distancept*signe)
                    {
                        if (abs(distancept) < 0.01)
                        {
                            mapdistpoint[poly.point(v)] = 0;
                        }
                        else
                        {
                            mapdistpoint[poly.point(v)] = distancept*signe;
                        }
                    }
                
                }
                else {
                    if (abs(distancept) < 0.01)
                    {
                        mapdistpoint[poly.point(v)] = 0;
                    }
                    else
                    {
                        mapdistpoint[poly.point(v)] = distancept*signe;
                    }

                }
                //-----------------------------------------------------------
            }
            score += dist;
        }
        else {
          
        }
    }
    return score/nombrepoint;
}


void loadBoth(string pathpcd,string pathply)
{
    
    if (!std::filesystem::exists(pathpcd)) {
        std::cerr << "Erreur : Le fichier PCD " << pathpcd << " n'existe pas.\n";
        return;
    }

    if (!std::filesystem::exists(pathply)) {
        std::cerr << "Erreur : Le fichier PLY " << pathply << " n'existe pas.\n";
        return;
    }

    if (pcl::io::loadPCDFile(pathpcd, *cloudXYZRG) == -1) {
        std::cerr << "Erreur : Impossible de charger le fichier PCD " << pathpcd << ".\n";
        return;
    }

    if (pcl::io::loadPCDFile(pathpcd, *cloud) == -1) {
        std::cerr << "Erreur : Impossible de charger le fichier PCD " << pathpcd << " dans cloud.\n";
        return;
    }

    if (pcl::io::loadPLYFile(pathply, cloud2) == -1) {
        std::cerr << "Erreur : Impossible de charger le fichier Ply pour pcl " << pathply << " dans cloud2.\n";
        return;
    }


    if (!CGAL::IO::read_PLY(pathply, test)) {
        std::cerr << "Erreur : Impossible de charger le fichier PLY " << pathply << ".\n";
        return;
    }

    std::cout << "Fichiers chargés avec succès.\n";
}


std::vector<Point> sample_mesh_faces(const mesh& poly, double density) {
    // Vecteur pour stocker les points échantillonnés
    std::vector<Point> sampled_points;

    // Échantillonner les faces du maillage
    CGAL::Polygon_mesh_processing::sample_triangle_mesh(
        poly,
        std::back_inserter(sampled_points),
        //CGAL::Polygon_mesh_processing::parameters::use_random_uniform_sampling(true),
        CGAL::Polygon_mesh_processing::parameters::number_of_points_on_faces(density)
        );
    ;
    cout<<"taille nuage de point "<< sampled_points.size() << endl;


    return sampled_points;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr convert_to_pcl_cloud(const std::vector<Point>& cgal_points) {
    PointCloud<PointXYZ>::Ptr pcl_cloud(new PointCloud<PointXYZ>);
    pcl_cloud->points.reserve(cgal_points.size());

    for (const auto& p : cgal_points) {
        pcl_cloud->points.emplace_back(p.x(), p.y(), p.z());
    }

    pcl_cloud->width = pcl_cloud->points.size();
    pcl_cloud->height = 1;  // Un seul rang de points
    pcl_cloud->is_dense = true;

    return pcl_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr addAndConvert(const mesh& poly, double density)
{
    return convert_to_pcl_cloud(sample_mesh_faces(poly,density));
}




PointCloud<pcl::PointXYZ>::Ptr align_point_clouds(
    PointCloud<pcl::PointXYZ>::Ptr cloud_source,
    PointCloud<pcl::PointXYZ>::Ptr cloud_target,
    double iteration,
    double transfoEpsi,
    double Fitness,
    double Maxcorres,
    double Iterransac,
    double outlier
) {
    // Initialiser l'algorithme ICP
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud_source);
    icp.setInputTarget(cloud_target);

    // Modifier les paramètres ICP
    icp.setMaximumIterations(iteration);
    icp.setTransformationEpsilon(transfoEpsi);
    icp.setEuclideanFitnessEpsilon(Fitness);
    icp.setMaxCorrespondenceDistance(Maxcorres);
    icp.setRANSACIterations(Iterransac);
    icp.setRANSACOutlierRejectionThreshold(outlier);

    // Nuage de points pour stocker le résultat aligné
    PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new PointCloud<pcl::PointXYZ>);

    // Exécuter l'alignement ICP en activant le multi-threading
#pragma omp parallel
    {
#pragma omp single
        std::cout << "Nombre de threads OpenMP utilisés : " << omp_get_num_threads() << std::endl;

#pragma omp single
        icp.align(*aligned_cloud);
    }

    // Vérifier la convergence
    if (icp.hasConverged()) {
        std::cout << "ICP converged with score: " << icp.getFitnessScore() << std::endl;
    }
    else {
        std::cerr << "ICP did not converge!" << std::endl;
    }

    return aligned_cloud;
}


pcl::PointCloud<pcl::Normal>::Ptr estimate_normals(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cl,
    double radius
) {
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;  // Utilisation d'OpenMP
    ne.setNumberOfThreads(10);  // Nombre de threads pour le calcul parallèle

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setInputCloud(cl);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius);
    ne.compute(*normals);

    return normals;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr compute_fpfh_features(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cl,
    pcl::PointCloud<pcl::Normal>::Ptr normals,
    double radius
) {
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;  // Utilisation d'OpenMP
    fpfh.setNumberOfThreads(10);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    fpfh.setInputCloud(cl);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(radius);
    fpfh.compute(*features);

    return features;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr align_with_descriptors(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features,
    double distanceSamplemin,
    double distanceCorres,
    double IterMax
) {
    std::cout << "on commence align" << std::endl;

    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;

    sac_ia.setInputSource(source_cloud);
    sac_ia.setSourceFeatures(source_features);
    sac_ia.setInputTarget(target_cloud);
    sac_ia.setTargetFeatures(target_features);
    sac_ia.setMinSampleDistance(distanceSamplemin);
    sac_ia.setMaxCorrespondenceDistance(distanceCorres);
    sac_ia.setMaximumIterations(IterMax);

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    sac_ia.align(*aligned_cloud);

    std::cout << "on fini align" << std::endl;
    if (sac_ia.hasConverged()) {
        std::cout << "SAC-IA converged with score: " << sac_ia.getFitnessScore() << std::endl;
    }
    else {
        std::cerr << "SAC-IA did not converge!" << std::endl;
    }

    return aligned_cloud;
}






void getResultat()
{
    double mindist = 50000;
    double maxdist = -50000;
   
    for (auto key : mapdistpoint)
    {
        
        if (key.second > maxdist)
        {
           
            maxdist = key.second;
            
        }
        if (key.second < mindist)
        {
            mindist = key.second;
            
        }

    }
    convertAndColorMesh(cloud2, mapdistpoint, mindist, maxdist);
    std::ofstream outfile("resultat_histogram.txt");
    cout <<"maxdist :" << maxdist << endl;
    cout<<"mindist :" << mindist << endl;
    outfile << "map des id de face et de la distance moyenne"  << std::endl;
    for (auto key : mapdistance)
    {

        outfile <<"id de la face "<<key.first<<" distance moyenne : "<<key.second<< std::endl;
       

    }

    outfile << "map de la densiter en fonction des faces " << std::endl;

    for (auto key : mapdens)
    {

        outfile << "id de la face " << key.first << " densité de point : " << key.second << std::endl;
      
    }

    for (auto key : mapdistpoint)
    {

        outfile << "id du point  " << key.first << " distance minimal: " << key.second << std::endl;

    }

    for (auto key : mapFaces)
    {
        outfile << "id de la face  " << key.first <<  std::endl;
        for (auto pt : key.second)
        {
            outfile << "id du point  " << pt << std::endl;
        }
    }

    outfile.close();

    

    // Sauvegarde du nuage en fichier PLY
    std::string filename = "colored_cloud.ply";
    if (pcl::io::savePLYFile(filename, cloud2) == 0) {
        std::cout << "Fichier PLY sauvegardé : " << filename << std::endl;
    }
    else {
        std::cerr << "Erreur lors de la sauvegarde du fichier PLY !" << std::endl;
    }
}


void Comparaison()
{
    string pathpcd;
    cout <<"entrer le path du fichier pcd"<< endl;
    cin >>pathpcd;

    string pathply;
    cout << "entrer le path du fichier ply" << endl;
    cin >> pathply;

    loadBoth(pathpcd, pathply);
    cloudsec = addAndConvert(test, 1);
    cout << "pcd et ply" << "defini" << endl;

    cout << "entrer le radius a utiliser pour le calcul de normal" << endl;
    double radiusnorm;
    cout << "la valeur qui a donner le meilleur resultat pour mes test est 2" << endl;
    cin >> radiusnorm;
    // Extraire les normales
    clock_t start = clock(); // Début du chrono
    auto source_normals = estimate_normals(cloud, radiusnorm);
    cout << "source normale" << "defini" << endl;
    auto target_normals = estimate_normals(cloudsec, radiusnorm);
    cout << "target normals" << "defini" << endl;
    clock_t end = clock(); // Fin du chrono
    double elapsed1 = double(end - start) / CLOCKS_PER_SEC; // Conversion en secondes
    std::cout << "Temps d'exécution : " << elapsed1 << " secondes\n";

    // Calculer les descripteurs FPFH

    cout << "entrer le radius a utiliser pour le calcul de feature du fpfh" << endl;
    double  radiusfeat;
    cout << "la valeur qui a donner le meilleur resultat est 5 plus cette valeur est grande plus le programme est long mais précis" << endl;

    cin >> radiusfeat;
    clock_t start1 = clock(); // Début du chrono
    auto source_features = compute_fpfh_features(cloud, source_normals,radiusfeat);
    cout << "source feature" << "defini" << endl;
    auto target_features = compute_fpfh_features(cloudsec, target_normals,radiusfeat);
    cout << "target feature" << "defini" << endl;
    clock_t end1 = clock(); // Fin du chrono
    double elapsed2 = double(end1 - start1) / CLOCKS_PER_SEC; // Conversion en secondes
    std::cout << "Temps d'exécution : " << elapsed2 << " secondes\n";
    // Aligner les nuages avec les descripteurs

    double distSam;
    double distCor;
    double iter;

    cout << "entrer les valeurs pour : la distance minimal de sample (0.09)" << endl;
    cin >> distSam;
    cout << "la distance de correspondance maximal (1)" << endl;
    cin >> distCor;
    cout << "le nombre d iteration de la correspondance (100)" << endl;
    cin >> iter;
    clock_t start2 = clock(); // Début du chrono
    auto initial_aligned_cloud = align_with_descriptors(cloud, source_features, cloudsec, target_features, distSam, distCor, iter);
    clock_t end2 = clock(); // Fin du chrono
    double elapsed3 = double(end2 - start2) / CLOCKS_PER_SEC; // Conversion en secondes
    std::cout << "Temps d'exécution : " << elapsed3 << " secondes\n";
    cout << "initial_aligned_cloud" << "defini" << endl;
    // Appliquer ICP pour raffiner l'alignement

    cout << "entrer les valeurs pour l'icp" << endl;

    cout << "entrer le nombre d iteration (1000)" << endl;
    double itericp;
    cin >> itericp;

    cout << "entrer le Critere d'arret (1e-8)" << endl;
    double crit;
    cin >> crit;

    cout << "entrer le second Critere d'arret (1e-5)" << endl;
    double secCrit;
    cin >> secCrit;

    cout << "entrer la distance maximal entre correspondance (1.5)" << endl;
    double distcorMax;
    cin >> distcorMax;

    cout << "entrer le nombre d'iteration de ransac (2000)" << endl;
    double iterrans;
    cin >> iterrans;

    cout << "entrer le seuil de rejet pour les outlier (1.4)" << endl;
    double seuil;
    cin >> seuil;
    auto final_aligned_cloud = align_point_clouds(initial_aligned_cloud, cloudsec,itericp,crit,secCrit,distcorMax,iterrans,seuil);
    cout << "final_aligned_cloud" << "defini" << transfo << endl;
    cloudXYZRG = convertToPointXYZRGB(final_aligned_cloud, 255, 0, 0);
    double score = 0;
    score = testFace(test, final_aligned_cloud);

    loadingPlyAndPcdView(test);
    getResultat();
}




int 
main ()
{
    /*
    loadBoth("D:/project/projettut/test/tete_VenusMilo.pcd", "D:/project/projettut/sphere.ply");
    */

    Comparaison();

    return 0;
}