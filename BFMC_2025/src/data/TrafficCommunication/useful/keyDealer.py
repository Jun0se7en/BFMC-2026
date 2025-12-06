import logging
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gen_key():
    """
    Generate a new RSA private key with a public exponent of 65537 and a key size of 2048 bits.
    Returns:
        private_key (cryptography.hazmat.backends.openssl.rsa._RSAPrivateKey): The generated private key.
    """
    logging.info("Generating a new RSA private key.")
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    logging.info("RSA private key generated.")
    return private_key

def save_private_key(pk, filename):
    """
    Save the private key to a file in PEM format.
    Args:
        pk (cryptography.hazmat.backends.openssl.rsa._RSAPrivateKey): The private key to save.
        filename (str): The name of the file to save the private key to.
    """
    logging.info(f"Saving private key to file: {filename}")
    pem = pk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(filename, "wb") as pem_out:
        pem_out.write(pem)
    logging.info(f"Private key saved to file: {filename}")

def save_public_key(pk, filename):
    """
    Save the public key to a file in PEM format.
    Args:
        pk (cryptography.hazmat.backends.openssl.rsa._RSAPublicKey): The public key to save.
        filename (str): The name of the file to save the public key to.
    """
    logging.info(f"Saving public key to file: {filename}")
    pem = pk.public_bytes(
        encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.PKCS1
    )
    with open(filename, "wb") as pem_out:
        pem_out.write(pem)
    logging.info(f"Public key saved to file: {filename}")

def load_private_key(filename):
    """
    Load a private key from a file in PEM format.
    Args:
        filename (str): The name of the file containing the private key.
    Returns:
        private_key (cryptography.hazmat.backends.openssl.rsa._RSAPrivateKey): The loaded private key.
    """
    logging.info(f"Loading private key from file: {filename}")
    with open(filename, "rb") as pem_in:
        pemlines = pem_in.read()
    private_key = load_pem_private_key(pemlines, None, default_backend())
    logging.info(f"Private key loaded from file: {filename}")
    return private_key

def load_public_key(filename):
    """
    Load a public key from a file in PEM format.
    Args:
        filename (str): The name of the file containing the public key.
    Returns:
        public_key (cryptography.hazmat.backends.openssl.rsa._RSAPublicKey): The loaded public key.
    """
    logging.info(f"Loading public key from file: {filename}")
    with open(filename, "rb") as pem_in:
        pemlines = pem_in.read()
    public_key = load_pem_public_key(pemlines, default_backend())
    logging.info(f"Public key loaded from file: {filename}")
    return public_key

def sign_data(private_key, plain_text):
    """
    Sign the given plain text using the private key.
    Args:
        private_key (cryptography.hazmat.backends.openssl.rsa._RSAPrivateKey): The private key to use for signing.
        plain_text (bytes): The plain text to sign.
    Returns:
        signature (bytes): The signature of the plain text.
    """
    logging.info("Signing data.")
    signature = private_key.sign(
        data=plain_text,
        padding=padding.PSS(
            mgf=padding.MGF1(hashes.MD5()), salt_length=padding.PSS.MAX_LENGTH
        ),
        algorithm=hashes.MD5(),
    )
    logging.info("Data signed.")
    return signature

def verify_data(public_key, plain_text, signature):
    """
    Verify the given plain text and signature using the public key.
    Args:
        public_key (cryptography.hazmat.backends.openssl.rsa._RSAPublicKey): The public key to use for verification.
        plain_text (bytes): The plain text to verify.
        signature (bytes): The signature to verify.
    Returns:
        is_signature_correct (bool): True if the signature is correct, False otherwise.
    """
    logging.info("Verifying data.")
    try:
        public_key.verify(
            signature=signature,
            data=plain_text,
            padding=padding.PSS(
                mgf=padding.MGF1(hashes.MD5()), salt_length=padding.PSS.MAX_LENGTH
            ),
            algorithm=hashes.MD5(),
        )
        is_signature_correct = True
        logging.info("Data verification successful.")
    except InvalidSignature:
        is_signature_correct = False
        logging.warning("Data verification failed.")
    return is_signature_correct