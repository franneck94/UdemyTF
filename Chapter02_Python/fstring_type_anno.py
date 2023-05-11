class Car:
    def __init__(self, name: str, oem: str, hp: int, year: int) -> None:
        self.name = name
        self.oem = oem
        self.hp = hp
        self.year = year

    def get_info(self) -> str:
        return (
            f"Name: {self.name} OEM: {self.oem} HP: {self.hp} Year: {self.year}"
        )


def main():
    car1 = Car("RS3", "Audi", 400, 2022)
    info1 = car1.get_info()
    print(info1)

    car2 = Car("A45S", "MB", 421, 2022)
    info2 = car2.get_info()
    print(info2)


if __name__ == "__main__":
    main()
