imageport cv2
imageport dlib
imageport numpy
imageport sys

path = "D:\TIGP\mm project\dlib-18.18\shape_predictor_68_face_landmarks.dat"
feather = 11
scale = 1 


face = list(range(17, 68))
mouth = list(range(48, 61))
eyebrowr = list(range(17, 22))
eyebrowl = list(range(22, 27))
eyer = list(range(36, 42))
eyel = list(range(42, 48))
nose = list(range(27, 35))
jaw = list(range(0, 17))


points_alignment = (eyebrowl + eyer + eyel +
                               eyebrowr + nose + mouth)


points_overlay = [
    eyel + eyer + eyebrowl + eyebrowr,
    nose + mouth,
]


colorblur_corr = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def g_lmarks(image):
    rects = detector(image, 1)
    
    if len(rects) > 1:
        raise ManyFaces
    if len(rects) == 0:
        raise NoFace

    return numpy.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def ann_lmarks(image, landmarks):
    image = image.copy()
    for idx, point in enumerate(landmarks):
        position = (point[0, 0], point[0, 1])
        cv2.putText(image, str(idx), position,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SimagePLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(image, position, 3, color=(0, 255, 255))
    return image

def hull_conv(image, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(image, points, color=color)

def face_mask(image, landmarks):
    image = numpy.zeros(image.shape[:2], dtype=numpy.float64)

    for group in points_overlay:
        hull_conv(image,
                         landmarks[group],
                         color=1)

    image = numpy.array([image, image, image]).transpositione((1, 2, 0))

    image = (cv2.GaussianBlur(image, (feather, feather), 0) > 0) * 1.0
    image = cv2.GaussianBlur(image, (feather, feather), 0)

    return image
    
def transformation(p1, p2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimageized.
    """
    
    p1 = p1.astype(numpy.float64)
    p2 = p2.astype(numpy.float64)

    c1 = numpy.mean(p1, axis=0)
    c2 = numpy.mean(p2, axis=0)
    p1 -= c1
    p2 -= c2

    s1 = numpy.std(p1)
    s2 = numpy.std(p2)
    p1 /= s1
    p2 /= s2

    U, S, Vt = numpy.linalg.svd(p1.T * p2)

    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def read_lmarks(fname):
    image = cv2.imageread(fname, cv2.imageREAD_COLOR)
    image = cv2.resize(image, (image.shape[1] * scale,
                         image.shape[0] * scale))
    s = g_lmarks(image)

    return image, s

def warp_image(image, M, dshape):
    output_image = numpy.zeros(dshape, dtype=image.dtype)
    cv2.warpAffine(image,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_image,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_image

def color_correction(image1, image2, landmarks1):
    amt_blur = colorblur_corr * numpy.linalg.norm(
                              numpy.mean(landmarks1[eyel], axis=0) -
                              numpy.mean(landmarks1[eyer], axis=0))
    amt_blur = int(amt_blur)
    if amt_blur % 2 == 0:
        amt_blur += 1
    image1_blur = cv2.GaussianBlur(image1, (amt_blur, amt_blur), 0)
    image2_blur = cv2.GaussianBlur(image2, (amt_blur, amt_blur), 0)

    
    image2_blur = image2_blur + 128 * (image2_blur <= 1.0)

    return (image2.astype(numpy.float64) * image1_blur.astype(numpy.float64) /
                                                image2_blur.astype(numpy.float64))

image1, landmarks1 = read_lmarks(sys.argv[1])
image2, landmarks2 = read_lmarks(sys.argv[2])

M = transformation(landmarks1[points_alignment],
                               landmarks2[points_alignment])

mask = face_mask(image2, landmarks2)
warped_mask = warp_image(mask, M, image1.shape)
combined_mask = numpy.max([face_mask(image1, landmarks1), warped_mask],
                          axis=0)

warped_image2 = warp_image(image2, M, image1.shape)
warped_corrected_image2 = color_correction(image1, warped_image2, landmarks1)

output_image = image1 * (1.0 - combined_mask) + warped_corrected_image2 * combined_mask

cv2.imagewrite('output.jpg', output_image)
