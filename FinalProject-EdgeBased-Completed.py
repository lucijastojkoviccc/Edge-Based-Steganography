import cv2
import numpy as np
import os

def detect_edges(image, method):
    if method == 'sobel':
        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Ky = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        Ix = cv2.filter2D(image, cv2.CV_64F, Kx)
        Iy = cv2.filter2D(image, cv2.CV_64F, Ky)
        G = np.sqrt(Ix ** 2 + Iy ** 2)
        edges = np.uint8(G)
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    
    # if method == 'sobel':
    #     grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    #     grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    #     edges = cv2.magnitude(grad_x, grad_y)
    #     edges = np.uint8(edges)

    elif method == 'prewitt':
        kernelx = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
        kernely = np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]])
        grad_x = cv2.filter2D(image, -1, kernelx)
        grad_y = cv2.filter2D(image, -1, kernely)
        edges = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))
        edges = np.uint8(edges)
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    elif method == 'canny':
        edges = cv2.Canny(image, 100, 200)


    elif method == 'log':
        log_kernel = np.array([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ], dtype=np.float32)

        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        edges = cv2.filter2D(blurred, -1, log_kernel)

        edges = np.uint8(np.absolute(edges))
        _, edges = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)

    # elif method == 'log':
    #     blurred = cv2.GaussianBlur(image, (3, 3), 0)
    #     edges = cv2.Laplacian(blurred, cv2.CV_64F)
    #     edges = np.uint8(np.absolute(edges))
    #     _, edges = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)


    else:
        raise ValueError("Nepoznata metoda detekcije ivica")
    #cv2.imwrite("detected_edges.png", edges)
    #print(f"Broj ivica (detektovano piksela): {np.sum(edges == 255)}")
    return edges

def embed_lsb(image, edges, message):
    binary_message = ''.join(format(ord(char), '08b') for char in message) + '11111111'
    stego_image = image.copy()
    idx = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if edges[y, x] == 255 and idx < len(binary_message):
                stego_image[y, x, 0] = (stego_image[y, x, 0] & ~1) | int(binary_message[idx])
                idx += 1
            if idx >= len(binary_message):
                break
        if idx >= len(binary_message):
            break
    return stego_image

def decode_lsb(image, edges, max_bytes=1024):
    binary_message = ""
    byte_limit = max_bytes * 8
    bit_count = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if edges[y, x] == 255:
                binary_message += str(image[y, x, 0] & 1)
                bit_count += 1
                if bit_count % 8 == 0:
                    char = chr(int(binary_message[-8:], 2))
                    if char == '\xFF':
                        return "".join(chr(int(binary_message[i:i + 8], 2)) for i in range(0, bit_count - 8, 8))
                if bit_count >= byte_limit:
                    break
        if bit_count >= byte_limit:
            break
    return "".join(chr(int(binary_message[i:i + 8], 2)) for i in range(0, bit_count, 8))


def save_image(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), image)

def main():
    mode = input("Izaberite režim rada (umetanje/dekodiranje): ").strip().lower()
    if mode == "umetanje":
        image_path = input("Unesite putanju do slike: ")
        image = cv2.imread(image_path)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_method = input("Unesite metodu detekcije ivica (sobel/canny/log/prewitt): ")
        edges = detect_edges(grayscale_image, method=edge_method)
        message = input("Unesite tajnu poruku: ")
        stego_image = embed_lsb(image, edges, message)
        save_folder = input("Unesite folder za čuvanje rezultata: ")
        stego_filename = input("Unesite naziv slike za skrivenu poruku: ")
        save_image(stego_image, save_folder, stego_filename)
        print(f"Slika sa skrivenom porukom je sačuvana kao {stego_filename} u folderu {save_folder}.")
    elif mode == "dekodiranje":
        image_path = input("Unesite putanju do slike za dekodiranje: ")
        image = cv2.imread(image_path)
        edge_method = input("Unesite metodu detekcije ivica (sobel/canny/log/prewitt): ")
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = detect_edges(grayscale_image, method=edge_method)
        message = decode_lsb(image, edges)
        print("Skrivena poruka je:", message)
    else:
        print("Nepoznat režim rada. Molimo izaberite 'umetanje' ili 'dekodiranje'.")

if __name__ == "__main__":
    main()