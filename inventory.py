T = input()
N = input()
inventory = []
inventory_quantity = []

for i in range(int(T)):
    for j in range(int(N)):
        item_input = input()
        items = item_input.split()
        inventory.append(items[0])
        inventory_quantity.append(int(items[1]))

M = input()
for i in range(int(M)):
    Modify_state = input()
    words = Modify_state.split()
    # print(words[2])
    if words[0] == 'ADD':
        flag = 0
        for j, element in enumerate(inventory):
            if element == words[1]:
                ch_item = inventory_quantity[j] + int(words[2])
                inventory_quantity[j] = ch_item
                print('ADDED item',inventory[j])
                flag = 1
                break
        if not flag and int(words[2]) > 0 and int(words[2]) < 101:
            inventory_quantity.append(int(words[2]))
            inventory.append(words[1])
            print('ADDED item',words[1])
    elif words[0] == 'DELETE':
        flag = 0
        for j, element in enumerate(inventory):
            if element == words[1]:
                flag = 1
                ch_item = inventory_quantity[j] - int(words[2])
                if ch_item < 0:
                    print("Item",words[1],"could not be DELETED")
                    break
                inventory_quantity[j] = ch_item
                print('DELETED Item',inventory[j])
                break
        if not flag:
            print("Item",words[1],"could not be DELETED")
sum = 0
for i in inventory_quantity:
    sum += i
print("Total Items in Inventory:",sum)
