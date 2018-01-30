#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>


using namespace cv;
using namespace std;

int main()
{

	// Objects
	Mat dianaOriginal;
	Mat dianaBinaria;
	Mat dianaRGB;
	Mat img_hsv;
	//Cargar la imagen en blanco y negro
	dianaOriginal = imread("definitiva.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (!dianaOriginal.data) {
		cout << "Error al cargar la imágen" << endl;
		return 1;
	}
	//Cargar la imagen en color
	dianaRGB = imread("definitiva.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	if (!dianaRGB.data) {
		cout << "Error al cargar la imágen" << endl;
		return 1;
	}

	//Las redimensiona para poder trabajar con ella
	Size size(1000, 1000);
	resize(dianaOriginal, dianaOriginal, size);
	resize(dianaRGB, dianaRGB, size);

	//DETECCIÓN DEL CENTRO DE LA DIANA ----------------------------------------------------------------------------------------------------------------------------------------

	//La convierte a binaria
	threshold(dianaOriginal, dianaBinaria, 70, 255, THRESH_BINARY);
	//Filtro para mejorar la detección de contornos
	GaussianBlur(dianaBinaria, dianaBinaria, Size(9, 9), 0, 0);

	//Detección de contornos.
	vector<vector<Point> > contours;
	findContours(dianaBinaria, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//Calcular momentos
	vector<Moments>momentos(contours.size());
	for (int i = 0; i <contours.size(); i++)
	{
		momentos[i] = moments(contours[i], false);
	}
	//Detectar centros
	vector<Point2f> centros(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		centros[i] = Point2f(momentos[i].m10 / momentos[i].m00, momentos[i].m01 / momentos[i].m00);
	}

	// Dibujo de contornos
	Mat dianaContornos;
	dianaOriginal.copyTo(dianaContornos);
	cvtColor(dianaContornos, dianaContornos, CV_GRAY2BGR);
	for (size_t k = 0; k < contours.size(); k++)
	{
		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dianaContornos, contours, k, color);
		circle(dianaContornos, centros[k], 4, color, -1, 8, 0);
	}

	//Detección del rectángulo entre todas las formas
	double areaRectangulo = 0;
	double areaTemporal;
	int posicionRectangulo;

	for (int i = 0; i < contours.size(); i++)
	{
		areaTemporal = contourArea(contours[i]);
		if (areaTemporal > areaRectangulo)
		{
			areaRectangulo = areaTemporal;
			posicionRectangulo = i;
		}
	}

	//Centro de la diana
	double centroX = centros[posicionRectangulo].x;
	double centroY = centros[posicionRectangulo].y;
	cout << "Centro de la diana" << endl;
	cout << centroX << " , " << centroY << endl;

	//Dibujo del rectángulo y su centro
	Mat dianaResultados;
	dianaOriginal.copyTo(dianaResultados);
	Scalar color(rand() & 255, rand() & 255, rand() & 255);
	cvtColor(dianaResultados, dianaResultados, CV_GRAY2BGR);
	drawContours(dianaRGB, contours, posicionRectangulo, color);
	circle(dianaRGB, centros[posicionRectangulo], 4, color, -1, 8, 0);

	//DETECCIÓN DISPAROS ----------------------------------------------------------------------------------------------------------------------------------------------------------


	//Conversión a HSV y división de cada canal
	cvtColor(dianaRGB, img_hsv, CV_BGR2HSV);

	Mat H(img_hsv.size(), CV_8UC1);
	Mat S(img_hsv.size(), CV_8UC1);
	Mat V(img_hsv.size(), CV_8UC1);
	Mat array_channels[] = { H,S,V };

	split(img_hsv, array_channels);

	threshold(H, H, 40, UCHAR_MAX, CV_THRESH_TOZERO);
	threshold(H, H, 75, UCHAR_MAX, CV_THRESH_BINARY_INV);
	threshold(S, S, 67, UCHAR_MAX, CV_THRESH_TOZERO);
	threshold(S, S, 255, UCHAR_MAX, CV_THRESH_BINARY_INV);
	threshold(V, V, 45, UCHAR_MAX, CV_THRESH_TOZERO);
	threshold(V, V, 189, UCHAR_MAX, CV_THRESH_BINARY_INV);

	Mat resultado;

	bitwise_and(H, S, resultado);
	bitwise_and(resultado, V, resultado);


	Mat R(img_hsv.size(), CV_8UC1);
	Mat G(img_hsv.size(), CV_8UC1);
	Mat B(img_hsv.size(), CV_8UC1);
	Mat array_channels_rgb[] = { R,G,B };

	split(dianaRGB, array_channels_rgb);


	threshold(R, R, 0, UCHAR_MAX, CV_THRESH_TOZERO);
	threshold(R, R, 100, UCHAR_MAX, CV_THRESH_BINARY_INV);


	threshold(G, G, 0, UCHAR_MAX, CV_THRESH_TOZERO);
	threshold(G, G, 100, UCHAR_MAX, CV_THRESH_BINARY_INV);


	threshold(B, B, 0, UCHAR_MAX, CV_THRESH_TOZERO);
	threshold(B, B, 100, UCHAR_MAX, CV_THRESH_BINARY_INV);

	int erosion_size_2 = 3;
	Mat element2 = getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(2 * erosion_size_2 + 1, 2 * erosion_size_2 + 1),
		cv::Point(erosion_size_2, erosion_size_2));


	int erosion_size = 6;
	Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));

	erode(G, G, element2);
	dilate(G, G, element2);


	Mat resultado1;


	bitwise_and(R, G, resultado1);

	bitwise_not(resultado1, resultado1);

	Mat resultadoF;


	bitwise_and(resultado, resultado1, resultadoF);



	//Aolicamos erosiones y dilataciones
	erode(resultadoF, resultadoF, element);
	dilate(resultadoF, resultadoF, element);
	erode(resultadoF, resultadoF, element);
	dilate(resultadoF, resultadoF, element);



	bitwise_and(resultadoF, B, resultadoF);

	blur(resultadoF, resultadoF, Size(3, 3));

	erode(resultadoF, resultadoF, element);
	dilate(resultadoF, resultadoF, element);



	int numeroDisparos;

	Mat labels;
	Mat stats;
	Mat centrosDisparos;

	int connectivity = 8;

	numeroDisparos = connectedComponentsWithStats(resultadoF, labels, stats, centrosDisparos, connectivity);

	cout << "Número de disparos detectados " << numeroDisparos - 1 << endl;


	RNG rng;
	for (int i = 1; i < stats.rows; i++)
	{
		Point cornerTopLeft(stats.at<int>(i, 0), stats.at<int>(i, 1));
		Point cornerBottomRight(stats.at<int>(i, 0) + stats.at<int>(i, 2), stats.at<int>(i, 1) + stats.at<int>(i, 3));
		Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rectangle(dianaRGB, cornerTopLeft, cornerBottomRight, color, 2);
		Point centro = Point(centrosDisparos.at<double>(i, 0), centrosDisparos.at<double>(i, 1));
		circle(dianaRGB, centro, 2, Scalar(0, 0, 255), CV_FILLED, 8, 0);

	}

	//CALCULO DE LA PUNTUACIÓN ---------------------------------------------------------------------------------------------------------------------------------------

	//Distancias en diana real:
	// 10 puntos: 0'6cm
	// 9 puntos: 0'6-1'4
	// 8 puntos: 1'4-2'2
	// 7 puntos: 2'2-3
	// 6 puntos: 3-3'8
	// 5 puntos: 3'8-4'6
	// 4 puntos: 4'6-5'4
	// 3 puntos: 5'4-6'2
	// 2 puntos: 6'2-7
	// 1 puntos: 7-7'8

	// Area diana real:
	// 17x17 = 289cm

	//Longitud X e Y del rectángulo
	double longitudRectangulo;
	longitudRectangulo = sqrt(areaRectangulo);

	//Regla de tres para obtener las distancias de puntuación en la imagen
	// distanciaLimiteImagen=(longitudRectanguloImagen*distanciaLimiteReal)/longitudRectanguloReal

	double uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve, diez;
	uno = (longitudRectangulo*7.8) / 17;
	dos = (longitudRectangulo * 7) / 17;
	tres = (longitudRectangulo*6.2) / 17;
	cuatro = (longitudRectangulo*5.4) / 17;
	cinco = (longitudRectangulo*4.6) / 17;
	seis = (longitudRectangulo*3.8) / 17;
	siete = (longitudRectangulo * 3) / 17;
	ocho = (longitudRectangulo*2.2) / 17;
	nueve = (longitudRectangulo*1.4) / 17;
	diez = (longitudRectangulo*0.6) / 17;

	double distancia;
	double distanciaX;
	double distanciaY;
	int puntuacion = 0;

	for (int i = 1; i < numeroDisparos; i++)
	{
		double centroDisparoX = centrosDisparos.at<double>(i, 0);
		double centroDisparoY = centrosDisparos.at<double>(i, 1);

		distanciaX = (centroX - centroDisparoX);
		distanciaY = (centroY - centroDisparoY);
		distancia = sqrt(pow(distanciaX, 2) + pow(distanciaY, 2));

		cout << endl << "Disparo numero " << i << endl;

		if (distancia < diez)
		{
			cout << "10 puntos" << endl;
			puntuacion = puntuacion + 10;
		}
		if (distancia > diez && distancia < nueve)
		{
			cout << "9 puntos" << endl;
			puntuacion = puntuacion + 9;
		}
		if (distancia > nueve && distancia < ocho)
		{
			cout << "8 puntos" << endl;
			puntuacion = puntuacion + 8;
		}
		if (distancia > ocho && distancia < siete)
		{
			cout << "7 puntos" << endl;
			puntuacion = puntuacion + 7;
		}
		if (distancia > siete && distancia < seis)
		{
			cout << "6 puntos" << endl;
			puntuacion = puntuacion + 6;
		}
		if (distancia > seis && distancia < cinco)
		{
			cout << "5 puntos" << endl;
			puntuacion = puntuacion + 5;
		}
		if (distancia > cinco && distancia < cuatro)
		{
			cout << "4 puntos" << endl;
			puntuacion = puntuacion + 4;
		}
		if (distancia > cuatro && distancia < tres)
		{
			cout << "3 puntos" << endl;
			puntuacion = puntuacion + 3;
		}
		if (distancia > tres && distancia < dos)
		{
			cout << "2 puntos" << endl;
			puntuacion = puntuacion + 2;
		}
		if (distancia > dos && distancia < uno)
		{
			cout << "1 punto" << endl;
			puntuacion = puntuacion + 1;
		}
	}
	cout << "Puntuacion total obtenida: " << puntuacion << endl;




	//IMPRESIÓN POR PANTALLA --------------------------------------------------------------------------------------------------------------------------------------

	//namedWindow("Diana Original", CV_WINDOW_AUTOSIZE);
	//namedWindow("Diana contornos", CV_WINDOW_AUTOSIZE);
	//namedWindow("Diana resultados", CV_WINDOW_AUTOSIZE);
	namedWindow("Diana RGB", CV_WINDOW_AUTOSIZE);

	//imshow("Diana Original", dianaOriginal);
	//imshow("Diana contornos", dianaContornos);
	//imshow("Diana resultados", dianaResultados);
	//imshow("original", resultadoF);
	imshow("Diana RGB", dianaRGB);

	waitKey(0);
	destroyAllWindows();

	//dianaOriginal.release();
	//dianaContornos.release();
	//dianaResultados.release();
	//resultadoF.release();
	dianaRGB.release();

	return 0;
}